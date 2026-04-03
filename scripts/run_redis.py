import os
from app.config.settings import settings

cmd = [
    "docker",
    "run",
    "-d",
    "--name",
    "my-redis",
    "-p",
    "6379:6379",
    "-v",
    "my-redis-data:/data",
    "redis",
    "redis-server",
]

if settings.STM_PERSISTENCE_MODE == "aof":
    cmd += ["--appendonly", "yes"]
elif settings.STM_PERSISTENCE_MODE == "rdb":
    cmd += [
        "--save",
        str(settings.STM_RDB_SAVE_INTERVAL),
        str(settings.STM_RDB_SAVE_CHANGES),
    ]
elif settings.STM_PERSISTENCE_MODE == "none":
    cmd += ["--save", ""]

print("Running Redis with command:", " ".join(cmd))
os.execvp(cmd[0], cmd)  # Replace current process
