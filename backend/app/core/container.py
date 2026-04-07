from __future__ import annotations

from backend.app.core.config import AppConfig
from backend.app.repositories.auth_audit_repository import AuthAuditRepository
from backend.app.repositories.filesystem.storage_repository import FilesystemStorageRepository
from backend.app.core.security import TokenStore
from backend.app.repositories.datasets_repository import DatasetsRepository
from backend.app.repositories.deployments_repository import DeploymentsRepository
from backend.app.repositories.hybrid_inspection_results_repository import HybridInspectionResultsRepository
from backend.app.repositories.inspection_results_repository import InspectionResultsRepository
from backend.app.repositories.models_repository import ModelsRepository
from backend.app.repositories.profiles_repository import ProfilesRepository
from backend.app.repositories.sqlserver.auth_audit_repository import SqlServerAuthAuditRepository
from backend.app.repositories.sqlserver.inspection_mirror_repository import SqlServerInspectionMirrorRepository
from backend.app.repositories.sqlserver.session_store import SqlServerTokenStore
from backend.app.repositories.sqlserver.users_repository import SqlServerUsersRepository
from backend.app.repositories.templates_repository import TemplatesRepository
from backend.app.repositories.training_repository import TrainingRepository
from backend.app.repositories.users_repository import UsersRepository
from backend.app.services.inspection_session import InspectionSessionService
from backend.app.services.sticker_inference import StickerInferenceService
from backend.app.services.template_runtime import TemplateRuntimeService
from backend.app.services.training import TrainingService


app_config = AppConfig()
filesystem_storage_repo = FilesystemStorageRepository()
users_repo = SqlServerUsersRepository(app_config) if app_config.sql_enabled else UsersRepository()
audit_repo = (
    SqlServerAuthAuditRepository(app_config)
    if app_config.sql_enabled
    else AuthAuditRepository()
)
templates_repo = TemplatesRepository()
deployments_repo = DeploymentsRepository()
profiles_repo = ProfilesRepository()
datasets_repo = DatasetsRepository()
models_repo = ModelsRepository()
training_repo = TrainingRepository()
local_inspection_results_repo = InspectionResultsRepository()
inspection_sql_mirror_repo = SqlServerInspectionMirrorRepository(app_config) if app_config.sql_enabled else None
inspection_results_repo = HybridInspectionResultsRepository(
    local_inspection_results_repo,
    inspection_sql_mirror_repo,
)

token_store = (
    SqlServerTokenStore(app_config, ttl_seconds=app_config.access_token_ttl_seconds)
    if app_config.sql_enabled
    else TokenStore(ttl_seconds=app_config.access_token_ttl_seconds)
)

template_runtime_service = TemplateRuntimeService(templates_repo, deployments_repo)
sticker_inference_service = StickerInferenceService(app_config, models_repo)
inspection_session_service = InspectionSessionService(
    template_runtime_service,
    profiles_repo,
    inspection_results_repo,
    sticker_inference_service,
)
training_service = TrainingService(training_repo)
