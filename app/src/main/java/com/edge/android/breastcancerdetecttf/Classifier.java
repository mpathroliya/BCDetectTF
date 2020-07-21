package com.edge.android.breastcancerdetecttf;


import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import static android.content.ContentValues.TAG;

public class Classifier {

    // Name of the file in the assets folder
    private static final String MODEL_PATH = "model.tflite";

    private Interpreter tflite;

    // Input Byte buffer
    private ByteBuffer inputBuffer = null;


    // output array
    private float[][] mnistOutput = null;


    //specify output size
    private static final int n = 2;

    //specify input size
    private static final int batchSize = 1;
    private static final int dimX = 224;
    private static final int dimY = 224;
    private static final int channels = 3;

    private static final int numOfBytesForFloat = 4;

    public Classifier(Activity activity) {
        try {
            // Define the TensorFlow Lite Interpreter with the model
            tflite = new Interpreter(loadModelFile(activity));

        } catch (IOException e) {
            Log.e(TAG, "IOException while loading the model");
        }
    }
    public void initialise(){
        // initialising the input buffer and output
        inputBuffer = ByteBuffer.allocateDirect(numOfBytesForFloat * batchSize * dimX * dimY * channels);
        inputBuffer.order(ByteOrder.nativeOrder());
        mnistOutput = new float[batchSize][n];
    }


    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    public int classify(Bitmap bitmap){
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
        }
        initialise();
        preprocess(bitmap);
        runInference();
        int predictedNumber = postProcess();
        return predictedNumber;
    }

    public void preprocess(Bitmap bitmap){
        if(bitmap ==null || inputBuffer ==null){
            return;
        }

        inputBuffer.rewind();

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        long startTime = SystemClock.uptimeMillis();

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels,0,width,0,0, width,height);

        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, " time taken to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    // this method runs inference
    public void runInference() {
        tflite.run(inputBuffer, mnistOutput);
    }

    public String resultString(){
        String res="";
        for (int i = 0; i < mnistOutput[0].length; i++){
            res += mnistOutput[0][i]+"\n";
        }
        return res;
    }


    // this method runs postprocess
    public int postProcess() {

        int maxIndex = -1;
        float maxvalue = 0.0f;

        for (int i = 0; i < mnistOutput[0].length; i++){

            if(mnistOutput[0][i] > maxvalue) {

                maxIndex = i;
                maxvalue = mnistOutput[0][i];
            }

        }
        return maxIndex;
    }


}