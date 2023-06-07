/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once


#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace legate_raft {
        namespace alloc {

            struct Allocator {

	        Allocator() = default;
                template<typename T>
                T *allocate_elements(size_t num_elements) {
                    return static_cast<T *>(allocate(sizeof(T) * num_elements));
                }

                virtual void *allocate(size_t bytes) = 0;

                virtual void deallocate(void *p) = 0;
            };

            class DeferredBufferAllocator : public Allocator {
            public:
                using Buffer = Legion::DeferredBuffer<int8_t, 1>;

            public:
                DeferredBufferAllocator() = default;

                DeferredBufferAllocator(Legion::Memory::Kind kind): target_kind(kind), Allocator() {}

                ~DeferredBufferAllocator() {
                    for(auto& pair : buffers) { pair.second.destroy(); }
		    buffers.clear();
		}

            public:
                void *allocate(size_t bytes) override {
                 if (bytes == 0) return nullptr;

  // Use 16-byte alignment
  bytes = (bytes + 15) / 16 * 16;
  Legion::Rect<1> bounds(Legion::Point<1>(0), Legion::Point<1>(bytes - 1));

  Buffer buffer(target_kind, Legion::Domain(bounds));
  void* ptr = buffer.ptr(0);
#ifdef DEBUG_LEGATE_RAFT
  assert(buffers.find(ptr) == buffers.end());
#endif
  buffers[ptr] = buffer;
  return ptr;
		}

                void deallocate(void *p) override {

			Buffer buffer;
  auto finder = buffers.find(p);
#ifdef DEBUG_LEGATE_RAFT
  assert(finder != buffers.end() || removed.find(p) != removed.end());
#endif
  if (finder == buffers.end()) return;
  buffer = finder->second;
  buffers.erase(finder);
  buffer.destroy();
		}

                bool is_popped(const void *p) const {
			return buffers.find(p) == buffers.end();
		}

                Buffer pop_allocation(const void *p) {
                      auto finder = buffers.find(p);
#ifdef DEBUG_LEGATE_RAFT
  assert(finder != buffers.end());
  removed.insert(finder->first);
#endif
  auto result = finder->second;
  buffers.erase(finder);
  return result;
		}

            private:
                Legion::Memory::Kind target_kind{Legion::Memory::Kind::SYSTEM_MEM};
                std::unordered_map<const void *, Buffer> buffers{};
#ifdef DEBUG_LEGATE_RAFT
                std::unordered_set<const void*> removed;
#endif
            };
        }

        struct DelegateAllocator : public rmm::mr::device_memory_resource {
            virtual void* delegate_allocation(std::size_t bytes, rmm::cuda_stream_view stream) = 0;
        };

        struct SingletonAllocator : public DelegateAllocator {
            SingletonAllocator() = default;

            SingletonAllocator(const std::pair<void*, size_t>& alloc) : idx(0), alloc(alloc) {}

            SingletonAllocator(void* ptr, size_t size) : idx(0), alloc(std::make_pair(ptr, size)) {}

            virtual bool supports_streams() const noexcept { return false; }

            virtual bool supports_get_mem_info() const noexcept { return false; }

            virtual void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
            {
                assert(idx++ == 0);
                return alloc.first;
            }

            virtual void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream)
            {
                // do nothing as we don't need to deallocate
            }

            virtual std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view stream) const
            {
                return std::make_pair(0, 0);
            }

            virtual void* delegate_allocation(std::size_t bytes, rmm::cuda_stream_view stream)
            {
                return do_allocate(bytes, stream);
            }

            unsigned idx;
            std::pair<void*, size_t> alloc;
        };

        struct DeferredBufferAllocator : public DelegateAllocator, alloc::DeferredBufferAllocator {
            using Buffer = Legion::DeferredBuffer<int8_t, 1>;

            DeferredBufferAllocator() : alloc::DeferredBufferAllocator(Legion::Memory::GPU_FB_MEM) {}
            virtual ~DeferredBufferAllocator() {}

            virtual bool supports_streams() const noexcept { return true; }

            virtual bool supports_get_mem_info() const noexcept { return false; }

            virtual void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
            {
                auto result = alloc::DeferredBufferAllocator::allocate(bytes);
                if (nullptr == result) return result;
#ifdef DEBUG_LEGATE_RAFT
                assert(stream_map.find(result) == stream_map.end());
#endif
                stream_map[result] = stream;
                return result;
            }

            virtual void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream)
            {
                if (!alloc::DeferredBufferAllocator::is_popped(p)) {
#ifdef DEBUG_LEGATE_RAFT
                    assert(stream_map.find(p) != stream_map.end());
#endif
                    // TODO: We really need deferred allocation and deallocation for CUDA kernels
                    //       to avoid blocking here.
                    stream_map[p].synchronize();
                    stream_map.erase(p);
                    alloc::DeferredBufferAllocator::deallocate(p);
                }
            }

            virtual std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view stream) const
            {
                return std::make_pair(0, 0);
            }

            virtual void* delegate_allocation(std::size_t bytes, rmm::cuda_stream_view stream)
            {
                return do_allocate(bytes, stream);
            }

            std::unordered_map<void*, rmm::cuda_stream_view> stream_map;
        };

        struct CompositeAllocator : public rmm::mr::device_memory_resource {
            using child = std::unique_ptr<DelegateAllocator>;

            CompositeAllocator() = default;

            virtual bool supports_streams() const noexcept { return false; }

            virtual bool supports_get_mem_info() const noexcept { return false; }

            virtual void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
            {
                assert(idx < children.size());
                return children[idx++]->delegate_allocation(bytes, stream);
            }

            virtual void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream)
            {
                // Do nothing
            }

            virtual std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view stream) const
            {
                return std::make_pair(0, 0);
            }

            template <typename Allocator, typename... Args>
            void add(Args... args)
            {
                children.push_back(std::make_unique<Allocator>(args...));
            }

            unsigned idx{0};
            std::vector<child> children{};
        };

}  // namespace legate_raft
