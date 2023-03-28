from legate.core import (
    Library,
    ResourceConfig,
    get_legate_runtime,
)
import os
from typing import Any

class UserLibrary(Library):
    def __init__(self, name: str) -> None:
        self.name = name
        self.shared_object: Any = None

    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from ../legate/raft.install_info import libpath
        return os.path.join(libpath, f"lib../legate/raft{self.get_library_extension()}")

    def get_c_header(self) -> str:
        from ../legate/raft.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "../legate/raft_perform_registration"

    def get_resource_configuration(self) -> ResourceConfig:
        assert self.shared_object is not None
        config = ResourceConfig()
        config.max_tasks = 1024
        config.max_mappers = 1
        config.max_reduction_ops = 8
        config.max_projections = 0
        config.max_shardings = 0
        return config

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass

user_lib = UserLibrary("../legate/raft")
user_context = get_legate_runtime().register_library(user_lib)
