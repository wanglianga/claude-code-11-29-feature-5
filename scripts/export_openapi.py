"""Write ``openapi.json`` from the running application definition.

The file is checked into the repository so the API contract is reviewable
without starting the service.  ``/docs`` serves the interactive Swagger UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402


def main() -> int:
    spec = app.openapi()
    out = ROOT / "openapi.json"
    out.write_text(json.dumps(spec, indent=2, sort_keys=False) + "\n")
    paths = len(spec["paths"])
    print(f"wrote {out} ({paths} paths, OpenAPI {spec['openapi']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
