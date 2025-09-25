from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from .api.routes import router as api_router
from .api.auth import router as auth_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Oratio Bias Detection API",
        version="0.1.0",
        default_response_class=ORJSONResponse,
    )

    app.include_router(api_router)
    app.include_router(auth_router)
    return app


app = create_app()


