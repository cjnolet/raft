legate_cpp_library_template(legate_raft TEMPLATE_SOURCES)

add_library(
        legate_raft

        # RAFT API Wrappers
        raft/histogram.cu
        raft/distance.cu
        raft/raft_kmeans.cu
#        raft/raft_knn.cu
#        raft/raft_knn_merge.cu

        # Legate tasks
        task/add.cc
        # task/compute_1nn.cc
        task/add_constant.cc
        task/bincount.cc
        task/categorize.cc
        task/convert.cc
        task/exp.cc
        task/fill.cc
        task/find_max.cc
        task/histogram.cc
        task/kmeans_fit_task.cc
        # task/knn_task.cc
        # task/knn_merge_task.cc
        task/log.cc
        task/matmul.cc
        task/mul.cc
        task/sum_over_axis.cc

        # Library templates
        ${TEMPLATE_SOURCES}
)

target_include_directories(legate_raft
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        INTERFACE
        $<INSTALL_INTERFACE:include>
        )

target_link_libraries(legate_raft PRIVATE legate::core raft::raft raft::compiled raft::distributed)


install(TARGETS legate_raft
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

include("${CMAKE_SOURCE_DIR}/helper_functions.cmake")
legate_add_cffi(legate_raft_cffi.h TARGET legate_raft PY_PATH legate/raft)
