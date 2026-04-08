import sys
import uvicorn
from openenv.core.env_server import create_fastapi_app
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import SupportTriageEnv, TriageAction, TriageObservation

def main():
    app = create_fastapi_app(SupportTriageEnv, TriageAction, TriageObservation)
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
