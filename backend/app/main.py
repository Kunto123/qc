from __future__ import annotations

from backend.app.factory import create_app


app = create_app()


if __name__ == "__main__":
    config = app.config["QC_SUITE"]
    app.run(host=config.host, port=config.port, debug=config.debug)

