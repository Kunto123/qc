from __future__ import annotations

from flask import Flask

from backend.app.api.auth_routes import auth_blueprint
from backend.app.api.calibration_routes import calibration_blueprint
from backend.app.api.dashboard_routes import dashboard_blueprint
from backend.app.api.deployment_routes import deployment_blueprint
from backend.app.api.inspection_routes import inspection_blueprint
from backend.app.api.template_routes import template_blueprint
from backend.app.api.workstation_routes import workstation_blueprint
from backend.app.core.config import AppConfig, ensure_data_dirs


def create_app() -> Flask:
    ensure_data_dirs()
    app = Flask(__name__)
    app.config["QC_SUITE"] = AppConfig()
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(template_blueprint)
    app.register_blueprint(deployment_blueprint)
    app.register_blueprint(inspection_blueprint)
    app.register_blueprint(dashboard_blueprint)
    app.register_blueprint(calibration_blueprint)
    app.register_blueprint(workstation_blueprint)

    @app.get("/health")
    def health():
        return {"status": "ok", "app": "qc-suite-python"}

    return app

