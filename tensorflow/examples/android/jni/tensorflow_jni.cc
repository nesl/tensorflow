/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/examples/android/jni/tensorflow_jni.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <jni.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "tensorflow/examples/android/jni/jni_utils.h"

using namespace tensorflow;

// Global variables that holds the Tensorflow classifier.
static std::unique_ptr<tensorflow::Session> session;

static std::vector<std::string> g_label_strings({"still", "walking", "running", "weightlifting"});
static bool g_compute_graph_initialized = false;
//static mutex g_compute_graph_mutex(base::LINKER_INITIALIZED);

// static int g_tensorflow_input_size;  // The image size for the mognet input.
// static int g_image_mean;  // The image mean.
static std::unique_ptr<StatSummarizer> g_stats;
static int g_tensorflow_n_steps;  // The n_steps for acc input.
static int g_tensorflow_n_input;  // The n_input for acc input.
static int g_tensorflow_n_classes;
static int g_tensorflow_n_hidden = 128;
static int g_tensorflow_n_layer = 3;

// For basic benchmarking.
static int g_num_runs = 0;
static int64 g_timing_total_us = 0;
static Stat<int64> g_frequency_start;
static Stat<int64> g_frequency_end;

// #ifdef LOG_DETAILED_STATS
// static const bool kLogDetailedStats = true;
// #else
// static const bool kLogDetailedStats = false;
// #endif

// Improve benchmarking by limiting runs to predefined amount.
// 0 (default) denotes infinite runs.
#ifndef MAX_NUM_RUNS
#define MAX_NUM_RUNS 0
#endif

// #ifdef SAVE_STEP_STATS
// static const bool kSaveStepStats = true;
// #else
// static const bool kSaveStepStats = false;
// #endif

static const bool kLogDetailedStats = false;
static const bool kSaveStepStats = false;

inline static int64 CurrentThreadTimeUs() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

JNIEXPORT jint JNICALL
TENSORFLOW_METHOD(initializeTensorflow)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager,
    jstring model, jstring labels,
    jint num_classes, jint n_steps, jint n_input) {

  LOG(INFO) << "In jni initializeTensorflow";
  LOG(INFO) << "n_steps=" << n_steps << ", n_input=" << n_input;

  g_num_runs = 0;
  g_timing_total_us = 0;
  g_frequency_start.Reset();
  g_frequency_end.Reset();

  //MutexLock input_lock(&g_compute_graph_mutex);
  if (g_compute_graph_initialized) {
    LOG(INFO) << "Compute graph already loaded. skipping.";
    return 0;
  }

  const int64 start_time = CurrentThreadTimeUs();

  const char* const model_cstr = env->GetStringUTFChars(model, NULL);
  // const char* const labels_cstr = env->GetStringUTFChars(labels, NULL);

  g_tensorflow_n_input = n_input;
  g_tensorflow_n_steps = n_steps; 
  g_tensorflow_n_classes = num_classes;

  LOG(INFO) << "Loading Tensorflow.";

  LOG(INFO) << "Making new SessionOptions.";
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  session.reset(tensorflow::NewSession(options));
  LOG(INFO) << "Session created.";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";

  AAssetManager* const asset_manager =
      AAssetManager_fromJava(env, java_asset_manager);
  LOG(INFO) << "Acquired AssetManager.";

  LOG(INFO) << "Reading file to proto: " << model_cstr;
  ReadFileToProto(asset_manager, model_cstr, &tensorflow_graph);

  g_stats.reset(new StatSummarizer(tensorflow_graph));

  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return -1;
  }

  std::vector<tensorflow::Tensor> tf_output;
  std::vector<std::string> output_names;

  // Print node count in the graph
  int node_count = tensorflow_graph.node_size();
  LOG(INFO) << node_count << " nodes in graph";

  // Iterate all nodes to restore weights
  for(int i = 0; i < node_count; i++) {
      auto n = tensorflow_graph.node(i);
      // If name contains "tf_weights", add to vector
      if(n.name().find("tf_weights") != std::string::npos) {
          LOG(INFO) << i << ":" << n.name();
          output_names.push_back(n.name());
      }
  }
  s = session->Run({}, output_names, {}, &tf_output);
  if (!s.ok()) {
    LOG(ERROR) << "Could not restore graph weights: " << s;
    return -1;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  LOG(INFO) << "Tensorflow graph loaded from: " << model_cstr;

  // // Read the label list
  // ReadFileToVector(asset_manager, labels_cstr, &g_label_strings);
  // LOG(INFO) << g_label_strings.size() << " label strings loaded from: "
  //           << labels_cstr;
  
  g_compute_graph_initialized = true;

  const int64 end_time = CurrentThreadTimeUs();
  LOG(INFO) << "Initialization done in " << (end_time - start_time) / 1000
            << "ms";

  return 0;
}

static int64 GetCpuSpeed() {
  string scaling_contents;
  ReadFileToString(nullptr,
                   "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
                   &scaling_contents);
  std::stringstream ss(scaling_contents);
  int64 result;
  ss >> result;
  return result;
}

JNIEXPORT jint JNICALL
TENSORFLOW_METHOD(classifyActivityAccRaw)(
    JNIEnv* env, jobject thiz, jint len, jfloatArray acc) {

  // Copy data into currFrame.
  jboolean iCopied = JNI_FALSE;
  jfloat* data = env->GetFloatArrayElements(acc, &iCopied);

  LOG(INFO) << "In classifyActivityAccRaw: " << len;

  // Create input tensor
  tensorflow::Tensor tensor_data(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, g_tensorflow_n_steps, g_tensorflow_n_input}));
  tensorflow::Tensor tensor_label(
    tensorflow::DT_FLOAT,
    tensorflow::TensorShape({1, g_tensorflow_n_classes}));
  tensorflow::Tensor tensor_state(
    tensorflow::DT_FLOAT,
    tensorflow::TensorShape({1, 2 * g_tensorflow_n_layer * g_tensorflow_n_hidden}));

  LOG(INFO) << "Tensorflow: Copying Data.";
  auto tensor_data_mapped = tensor_data.tensor<float, 3>();
  for (int i = 0; i < g_tensorflow_n_steps; ++i) {
    for (int j = 0; j < g_tensorflow_n_input; ++j) {
      tensor_data_mapped(0, i, j) = data[i * g_tensorflow_n_input + j];
    }
  }
  auto tensor_label_mapped = tensor_label.tensor<float, 2>();
  tensor_label_mapped(0, 0) = 1.0;
 
  std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors(
      {{"tf_data", tensor_data}, {"tf_state", tensor_state}, {"tf_label", tensor_label}});

  VLOG(0) << "Start computing.";
  std::vector<tensorflow::Tensor> output_tensors;

  tensorflow::Status s;
  int64 start_time, end_time;

  start_time = CurrentThreadTimeUs();
  s = session->Run(input_tensors, {"tf_rnn_5"}, {}, &output_tensors);
  end_time = CurrentThreadTimeUs();
  
  const int64 elapsed_time_inf = end_time - start_time;
  g_timing_total_us += elapsed_time_inf;
  g_num_runs++;
  VLOG(0) << "End computing. Ran in " << elapsed_time_inf / 1000 << "ms ("
          << (g_timing_total_us / g_num_runs / 1000) << "ms avg over "
          << g_num_runs << " runs)";

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return -1;
  }

  int res = -1;
  int max = 0;
  VLOG(0) << "Reading output tensor";
  auto tensor_output_mapped = output_tensors[0].tensor<float, 2>();
  for(int i = 0; i < g_tensorflow_n_classes; ++i) {
    LOG(INFO) << "class " << i << ": " << tensor_output_mapped(0, i);
    if (tensor_output_mapped(0, i) > max) {
      max = tensor_output_mapped(0, i);
      res = i;
    }
  }

  return res;
}
