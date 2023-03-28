
legate_cpp_library_template(legate_raft TEMPLATE_SOURCES)

add_library(
        legate_raft

        # RAFT API wrappers
        raft/raft_knn.cu

        # Legate tasks
        legate_tasks/knn_task.cc

        # Library templates
        ${TEMPLATE_SOURCES}
)

target_include_directories(legate_raft
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        INTERFACE
        $<INSTALL_INTERFACE:include>
        )

target_link_libraries(legate_raft PRIVATE legate::core raft::raft raft::compiled)