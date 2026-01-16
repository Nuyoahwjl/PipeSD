import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def deduplicate_records(records: Iterable[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    deduped: List[Dict] = []
    removed = 0

    for record in records:
        if not isinstance(record, dict):
            deduped.append(record)
            continue

        task_id = record.get("task_id")
        if task_id in seen:
            removed += 1
            continue

        seen.add(task_id)
        deduped.append(record)

    return deduped, removed


def process_file(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skipping {path} ({exc})")
        return 0

    if not isinstance(data, list):
        print(f"Skipping {path} (expected top-level list)")
        return 0

    deduped, removed = deduplicate_records(data)
    if removed == 0:
        return 0

    with path.open("w", encoding="utf-8") as file:
        json.dump(deduped, file, indent=4)
        file.write("\n")

    print(f"{path}: removed {removed} duplicate entries")
    return removed


def main() -> None:
    root = Path(__file__).parent
    total_removed = 0

    for json_file in sorted(root.rglob("*.json")):
        total_removed += process_file(json_file)

    print(f"Finished. Removed {total_removed} duplicate entries in total.")


if __name__ == "__main__":
    main()
