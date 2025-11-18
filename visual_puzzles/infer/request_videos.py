import argparse
import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import requests


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Batch video requester for PuzzleVQA tasks")
	parser.add_argument("--model", required=True, help="Model name to query")
	parser.add_argument("--base_url", required=True, help="Inference service base URL")
	parser.add_argument("--api_key", help="API key (optional, falls back to dotenv)")
	parser.add_argument("--tasks", nargs="+", required=True, help="Task names to evaluate")
	parser.add_argument("--data_root", default="data", help="Directory containing task folders")
	parser.add_argument("--output_root", required=True, help="Directory where results are stored")
	parser.add_argument("--no_images", action="store_true", help="Do not include images in requests")
	parser.add_argument("--threads", type=int, default=4, help="Thread pool size for requests/downloads")
	parser.add_argument("--max_request_attempts", type=int, default=1, help="Maximum attempts per video request")
	parser.add_argument(
		"--request_attempt_delay",
		type=float,
		default=0.0,
		help="Seconds to wait between request attempts",
	)
	return parser.parse_args()


def setup_logging() -> None:
	logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


def image_to_base64(image_path: Path) -> str:
	with image_path.open("rb") as image_file:
		return base64.b64encode(image_file.read()).decode("utf-8")


def load_dataset(dataset_path: Path) -> List[Dict[str, object]]:
	with dataset_path.open("r", encoding="utf-8") as fp:
		data = json.load(fp)
	if not isinstance(data, list):
		raise ValueError(f"Dataset at {dataset_path} is not a list of entries")
	for entry in data:
		if isinstance(entry, dict):
			entry.pop("video_path", None)
	return data


def resolve_image_path(entry: Dict[str, object], dataset_dir: Path) -> Optional[Path]:
	image_value = entry.get("image") if isinstance(entry, dict) else None
	if not image_value or not isinstance(image_value, str):
		return None
	candidate = dataset_dir / image_value
	return candidate if candidate.is_file() else None


def build_messages(text_prompt: str, image_base64: Optional[str]) -> List[Dict[str, object]]:
	content: List[Dict[str, object]] = [{"type": "text", "text": text_prompt}]
	if image_base64:
		content.append(
			{
				"type": "image_url",
				"image_url": {"url": f"data:image/png;base64,{image_base64}"},
			}
		)
	return [{"role": "user", "content": content}]


def request_entry(
	client: OpenAI,
	model_name: str,
	entry: Dict[str, object],
	dataset_dir: Path,
	include_image: bool,
) -> str:
	entry_id = entry.get("id") or entry.get("index") or entry.get("question_id")
	prefix = f"Entry {entry_id}" if entry_id is not None else "Entry"

	prompt_value = entry.get("prompt") if isinstance(entry, dict) else None
	if not prompt_value or not isinstance(prompt_value, str):
		raise ValueError(f"{prefix}: missing prompt field")

	image_b64: Optional[str] = None
	if include_image:
		image_path = resolve_image_path(entry, dataset_dir)
		if image_path:
			image_b64 = image_to_base64(image_path)
		else:
			logging.warning("%s: image path %r could not be resolved", prefix, entry.get("image"))

	messages = build_messages(prompt_value, image_b64)

	logging.debug("%s: sending request", prefix)
	response = client.chat.completions.create(model=model_name, messages=messages)
	content = response.choices[0].message.content or ""

	pattern = r"\(https?://[^\s)]+\)"
	match = re.search(pattern, content)
	if not match:
		raise ValueError(f"{prefix}: response did not include a video URL")

	video_url = match.group(0)[1:-1]
	if not video_url:
		raise ValueError(f"{prefix}: unable to parse video URL from response")

	return video_url


def request_entries_for_indices(
	entries: List[Dict[str, object]],
	indices: List[int],
	client: OpenAI,
	model_name: str,
	dataset_dir: Path,
	include_image: bool,
	threads: int,
	attempt_label: str,
) -> Tuple[Dict[int, str], Dict[int, str]]:
	results: Dict[int, str] = {}
	errors: Dict[int, str] = {}

	if not indices:
		return results, errors

	with ThreadPoolExecutor(max_workers=threads) as executor:
		future_to_index = {
			executor.submit(
				request_entry,
				client,
				model_name,
				entries[idx],
				dataset_dir,
				include_image,
			): idx
			for idx in indices
		}
		progress = tqdm(total=len(future_to_index), desc=attempt_label, unit="req")
		try:
			for future in as_completed(future_to_index):
				idx = future_to_index[future]
				try:
					results[idx] = future.result()
				except Exception as exc:  # noqa: BLE001
					errors[idx] = str(exc)
				finally:
					progress.update()
		finally:
			progress.close()

	return results, errors


def download_video(video_url: str, target_path: Path) -> None:
	response = requests.get(video_url, timeout=60)
	response.raise_for_status()
	target_path.write_bytes(response.content)


def make_video_filename(entry: Dict[str, object], idx: int) -> str:
	entry_id = entry.get("id") or entry.get("index") or entry.get("question_id")
	if isinstance(entry_id, str):
		safe = re.sub(r"[^0-9A-Za-z-_]+", "_", entry_id).strip("_")
		return f"{safe or f'entry_{idx}'}.mp4"
	return f"entry_{idx}.mp4"


def download_videos(
	entries: List[Dict[str, object]],
	task_dirs: Dict[str, Path],
	index_to_url: Dict[int, str],
	threads: int,
	stage_label: str,
) -> List[Tuple[int, str]]:
	if not index_to_url:
		return []

	def handle_download(idx: int, video_url: str) -> None:
		entry = entries[idx]
		video_filename = make_video_filename(entry, idx)
		video_path = task_dirs["videos_dir"] / video_filename
		download_video(video_url, video_path)
		entry["video_path"] = str(video_path.relative_to(task_dirs["task_dir"]))

	download_errors: List[Tuple[int, str]] = []

	with ThreadPoolExecutor(max_workers=threads) as executor:
		future_to_idx = {
			executor.submit(handle_download, idx, video_url): idx for idx, video_url in index_to_url.items()
		}
		progress = tqdm(total=len(future_to_idx), desc=f"{stage_label} | downloads", unit="file")
		try:
			for future in as_completed(future_to_idx):
				idx = future_to_idx[future]
				try:
					future.result()
				except Exception as exc:  # noqa: BLE001
					entries[idx].pop("video_path", None)
					download_errors.append((idx, str(exc)))
				finally:
					progress.update()
		finally:
			progress.close()

	return download_errors


def ensure_task_dirs(output_root: Path, task_name: str) -> Dict[str, Path]:
	task_dir = output_root / task_name
	task_dir.mkdir(parents=True, exist_ok=True)
	videos_dir = task_dir / "videos"
	videos_dir.mkdir(parents=True, exist_ok=True)
	return {
		"task_dir": task_dir,
		"videos_dir": videos_dir,
		"result_path": task_dir / "video_result.json",
	}


def to_absolute_path(path_value: Optional[str], base_dir: Path) -> Optional[str]:
	if not path_value or not isinstance(path_value, str):
		return path_value
	candidate = Path(path_value)
	if not candidate.is_absolute():
		candidate = base_dir / candidate
	return str(candidate.resolve())


def write_results(
	entries: List[Dict[str, object]],
	task_dirs: Dict[str, Path],
	stage_label: str,
	dataset_dir: Path,
) -> None:
	result_entries: List[Dict[str, object]] = []
	for entry in entries:
		if not isinstance(entry, dict):
			result_entries.append(entry)
			continue
		entry_copy = dict(entry)
		image_abs = to_absolute_path(entry_copy.get("image"), dataset_dir)
		if image_abs:
			entry_copy["image"] = image_abs
		solution_abs = to_absolute_path(entry_copy.get("solution_image_path"), dataset_dir)
		if solution_abs:
			entry_copy["solution_image_path"] = solution_abs
		video_abs = to_absolute_path(entry_copy.get("video_path"), task_dirs["task_dir"])
		if video_abs:
			entry_copy["video_path"] = video_abs
		result_entries.append(entry_copy)

	result_path = task_dirs["result_path"]
	with result_path.open("w", encoding="utf-8") as fp:
		json.dump(result_entries, fp, indent=2, ensure_ascii=False)
	logging.info("%s: results written to %s", stage_label, result_path)


def log_stage_errors(stage_name: str, errors: List[Tuple[int, str]]) -> None:
	for idx, message in errors:
		logging.error("Entry %d %s failed: %s", idx, stage_name, message)


def process_task(
	task_name: str,
	entries: List[Dict[str, object]],
	dataset_path_obj: Path,
	task_dirs: Dict[str, Path],
	request_client: OpenAI,
	request_model: str,
	include_image: bool,
	threads: int,
	request_max_attempts: int,
	request_attempt_delay: float,
) -> None:
	dataset_dir = dataset_path_obj.parent
	stage_label = task_name

	pending_indices = list(range(len(entries)))
	last_request_errors: Dict[int, str] = {}
	max_attempts = max(1, request_max_attempts)
	aggregate_download_errors: List[Tuple[int, str]] = []

	write_results(entries, task_dirs, stage_label, dataset_dir)

	for attempt in range(1, max_attempts + 1):
		if not pending_indices:
			break

		logging.info(
			"%s: requesting %d videos (attempt %d/%d)",
			task_name,
			len(pending_indices),
			attempt,
			max_attempts,
		)
		attempt_label = f"{stage_label} | requests {attempt}/{max_attempts}"
		attempt_results, attempt_errors = request_entries_for_indices(
			entries,
			pending_indices,
			request_client,
			request_model,
			dataset_dir,
			include_image,
			threads,
			attempt_label,
		)

		for idx, error_message in attempt_errors.items():
			last_request_errors[idx] = error_message

		success_index_to_url = {idx: attempt_results[idx] for idx in attempt_results}

		if success_index_to_url:
			logging.info(
				"%s: downloading %d videos from attempt %d",
				task_name,
				len(success_index_to_url),
				attempt,
			)
			download_errors = download_videos(
				entries,
				task_dirs,
				success_index_to_url,
				threads,
				stage_label,
			)
			if download_errors:
				log_stage_errors("download", download_errors)
				aggregate_download_errors.extend(download_errors)
		else:
			logging.info("%s: no successful requests in attempt %d", task_name, attempt)

		write_results(entries, task_dirs, stage_label, dataset_dir)

		pending_indices = [idx for idx in pending_indices if idx not in success_index_to_url]

		if not pending_indices:
			break

		if attempt < max_attempts:
			logging.info(
				"%s: %d videos pending after attempt %d",
				task_name,
				len(pending_indices),
				attempt,
			)
			if request_attempt_delay > 0:
				logging.info(
					"%s: waiting %.2f seconds before next request attempt",
					stage_label,
					request_attempt_delay,
				)
				time.sleep(request_attempt_delay)

	if pending_indices:
		for idx in pending_indices:
			error_message = last_request_errors.get(idx, "Unknown error")
			logging.error("Entry %d request failed: %s", idx, error_message)
		logging.warning(
			"%s: %d entries failed after %d attempts",
			task_name,
			len(pending_indices),
			max_attempts,
		)
	else:
		logging.info("%s: all %d entries succeeded", task_name, len(entries))

	if aggregate_download_errors:
		logging.warning(
			"%s: %d download failures encountered",
			task_name,
			len(aggregate_download_errors),
		)


def main() -> None:
	args = parse_args()
	setup_logging()

	if not args.api_key:
		load_dotenv()
	api_key = args.api_key or os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("API key not provided and OPENAI_API_KEY not set")

	include_image = not args.no_images
	output_root = Path(args.output_root)
	output_root.mkdir(parents=True, exist_ok=True)

	client = OpenAI(api_key=api_key, base_url=args.base_url)

	threads = max(args.threads, 1)
	data_root = Path(args.data_root)

	for task_name in args.tasks:
		dataset_path = data_root / task_name / "data.json"
		if not dataset_path.is_file():
			logging.error("Task %s: dataset not found at %s", task_name, dataset_path)
			continue

		task_dirs = ensure_task_dirs(output_root, task_name)

		logging.info("Loading task %s from %s", task_name, dataset_path)
		try:
			entries = load_dataset(dataset_path)
		except Exception as exc:  # noqa: BLE001
			logging.exception("Failed to load task %s: %s", task_name, exc)
			continue

		try:
			process_task(
				task_name,
				entries,
				dataset_path,
				task_dirs,
				client,
				args.model,
				include_image,
				threads,
				max(args.max_request_attempts, 1),
				max(args.request_attempt_delay, 0.0),
			)
		except Exception as exc:  # noqa: BLE001
			logging.exception("%s: unexpected error: %s", task_name, exc)


if __name__ == "__main__":
	main()

