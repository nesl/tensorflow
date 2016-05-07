/*
 * Copyright 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

public class CameraActivity extends Activity implements View.OnClickListener, SensorEventListener {
    private static final String TAG = "SensorDemo";
    private TextView mAccViewX, mAccViewY, mAccViewZ;

    private static final int BUFFER_SIZE = 50 * 3;
    private SensorManager sensorManager;
    private static float[] accData = new float[BUFFER_SIZE + 1];
    private static int accCount = 0;

    private static final String MODEL_FILE = "file:///android_asset/example_graph_1.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";
    private final TensorflowClassifier tensorflow = new TensorflowClassifier();

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        // Set button listeners for start/stop
        Button button1 = (Button) findViewById(R.id.button1);
        Button button2 = (Button) findViewById(R.id.button2);
        if (button1 != null) {
            button1.setOnClickListener(this);
        }
        if (button2 != null) {
            button2.setOnClickListener(this);
        }

        // Get handle to text fields
        mAccViewX = (TextView) findViewById(R.id.textView1);
        mAccViewY = (TextView) findViewById(R.id.textView2);
        mAccViewZ = (TextView) findViewById(R.id.textView3);

        // Initialize tensorflow model
        tensorflow.initializeTensorflow(
                getAssets(), MODEL_FILE, LABEL_FILE, 4, 10, 10);
    }

    @Override
    public void onClick(View v) {
        // Start sensor sampling
        if (v.getId() == R.id.button1) {
            sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
            Sensor mAcc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            sensorManager.registerListener(this, mAcc, SensorManager.SENSOR_DELAY_GAME);
            Log.i(TAG, "start clicked");
        }

        // Stop sensor sampling
        if (v.getId() == R.id.button2) {
            Log.i(TAG, "stop clicked");
            sensorManager.unregisterListener(this);
            sensorManager = null;
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        // Log.i(TAG, "sensor data received");
        mAccViewX.setText("Acc X = " + event.values[0]);
        mAccViewY.setText("Acc Y = " + event.values[1]);
        mAccViewZ.setText("Acc Z = " + event.values[2]);

        // Save data in buffer
        accData[accCount] = event.values[0];
        accData[accCount + 1] = event.values[1];
        accData[accCount + 2] = event.values[2];
        accCount += 3;

        // Classify once we have enough data
        if (accCount >= BUFFER_SIZE) {
            Log.i(TAG, "buffer full, len =  " + accCount);
            float[] data = new float[accCount];
            for (int i = 0; i < accCount; i++) {
                data[i] = accData[i];
                accData[i] = 0;
            }
            accCount = 0;
            final int res = tensorflow.recognizeActivity(data.length, data);
            Log.i(TAG, "Classification result = " + res);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}
