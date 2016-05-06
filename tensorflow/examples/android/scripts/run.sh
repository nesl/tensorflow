bazel build //tensorflow/examples/android:tensorflow_demo
adb install -r -g ../../../bazel-bin/tensorflow/examples/android/tensorflow_demo.apk
