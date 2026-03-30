cmake_minimum_required(VERSION 3.20)
project(mini_llm_gru C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4 /O2)
else()
  add_compile_options(-Wall -Wextra -O3)
endif()

set(MINI_LLM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(MINI_LLM_UI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ui")

add_library(mini_llm
  "${MINI_LLM_SRC_DIR}/util.c"
  "${MINI_LLM_SRC_DIR}/tokenizer_byte.c"
  "${MINI_LLM_SRC_DIR}/dataset_csv.c"
  "${MINI_LLM_SRC_DIR}/model_gru.c"
  "${MINI_LLM_SRC_DIR}/plot_png.c"
)

target_include_directories(mini_llm PUBLIC "${MINI_LLM_SRC_DIR}")

add_executable(train_main
  "${MINI_LLM_SRC_DIR}/train_main.c"
)
target_link_libraries(train_main PRIVATE mini_llm)

add_executable(chat_server_main
  "${MINI_LLM_SRC_DIR}/chat_server_main.c"
)
target_link_libraries(chat_server_main PRIVATE mini_llm ws2_32)

