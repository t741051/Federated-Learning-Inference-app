package com.example.fl_app;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import android.Manifest;
import android.content.pm.PackageManager;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private static final int INPUT_IMAGE_SIZE = 75;
    private static final int NUM_CLASSES = 17;

    private ImageView imageView;
    private ImageView vector;
    private TextView resultTextView;
    private Button inferenceButton;

    private Button cameraButton;

    private Interpreter tfliteInterpreter;
    private ImageProcessor imageProcessor;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputBuffer;
    private List<String> classLabels;

    public static final int CAMERA_PERM_CODE = 101;
    public static final int CAMERA_REQUEST_CODE = 102;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        resultTextView = findViewById(R.id.resultTextView);
        inferenceButton = findViewById(R.id.inferenceButton);
        cameraButton = findViewById(R.id.cameraButton);
        vector = findViewById(R.id.vector);


        inferenceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openImagePicker();
            }
        });

        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                askCameraPermissions();
            }
        });

        try {
            tfliteInterpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }

        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
                .build();

        inputImageBuffer = new TensorImage(DataType.FLOAT32);
        outputBuffer = TensorBuffer.createFixedSize(new int[]{1, NUM_CLASSES}, DataType.FLOAT32);

        classLabels = loadLabels();
        // 修改布局参数中的位置信息


    }
    private void askCameraPermissions() {
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,new String[] {Manifest.permission.CAMERA}, CAMERA_PERM_CODE);
        }else {
            openCamera();
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERM_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "Camera Permission is Required to Use camera.", Toast.LENGTH_SHORT).show();
            }
        }
    }
    private void openCamera() {
        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(camera, CAMERA_REQUEST_CODE);
    }
    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                imageView.setImageBitmap(bitmap);
                performInference(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(requestCode == CAMERA_REQUEST_CODE){
            Bitmap bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
            performInference(bitmap);

        }
    }

    private void performInference(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);

// 图像处理和转换
        int width = resizedBitmap.getWidth();
        int height = resizedBitmap.getHeight();
        float[] pixels = new float[width * height * 3];  // 3 channels (RGB)
        int index = 0;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = resizedBitmap.getPixel(i, j);
                // 获取 RGB 三个通道的像素值
                int r = Color.red(pixel);
                int g = Color.green(pixel);
                int b = Color.blue(pixel);
                // 归一化像素值到 [0, 1]
                float normalizedR = r / 255.0f;
                float normalizedG = g / 255.0f;
                float normalizedB = b / 255.0f;
                // 存储归一化后的像素值到数组
                pixels[index++] = normalizedR;
                pixels[index++] = normalizedG;
                pixels[index++] = normalizedB;
            }
        }

// 将处理后的图像加载到输入张量
        inputImageBuffer.load(pixels, new int[]{1, width, height, 3}); // Assuming NHWC tensor format

// 执行推理
        tfliteInterpreter.run(inputImageBuffer.getBuffer(), outputBuffer.getBuffer().rewind());

// 获取推理结果
        float[] results = outputBuffer.getFloatArray();

// 处理推理结果
        String inferenceResult = processInferenceResult(results);

// 显示推理结果
        resultTextView.setText("推論結果: " + inferenceResult);

// 更改箭頭位置及角度
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) vector.getLayoutParams();
        float rotationAngle = 0;
        switch (inferenceResult) {
            case "1-1":
                layoutParams.leftMargin = 220;
                layoutParams.bottomMargin = 260;
                rotationAngle = 270f; // 设置旋转角度为45度
                break;
            case "1-2":
                layoutParams.leftMargin = 250;
                layoutParams.bottomMargin = 300;
                break;
            case "2-1":
                layoutParams.leftMargin = 250;
                layoutParams.bottomMargin = 370;
                rotationAngle = 180f; // 设置旋转角度为45度
                break;
            case "2-2":
                layoutParams.leftMargin = 250;
                layoutParams.bottomMargin = 400;
                rotationAngle = 90f; // 设置旋转角度为45度
                break;
            case "3-1":
                layoutParams.leftMargin = 300;
                layoutParams.bottomMargin = 390;
                rotationAngle = 270f; // 设置旋转角度为45度
                break;
            case "3-2":
                layoutParams.leftMargin = 300;
                layoutParams.bottomMargin = 390;
                rotationAngle = 90f; // 设置旋转角度为45度
                break;
            case "4-1":
                layoutParams.leftMargin = 550;
                layoutParams.bottomMargin = 390;
                rotationAngle = 270f; // 设置旋转角度为45度
                break;
            case "4-2":
                layoutParams.leftMargin = 550;
                layoutParams.bottomMargin = 390;
                rotationAngle = 90f; // 设置旋转角度为45度
                break;
            case "5-1":
                layoutParams.leftMargin = 700;
                layoutParams.bottomMargin = 390;
                rotationAngle = 270f; // 设置旋转角度为45度
                break;
            case "5-2":
                layoutParams.leftMargin = 700;
                layoutParams.bottomMargin = 390;
                rotationAngle = 90f; // 设置旋转角度为45度
                break;
            case "6-1":
                layoutParams.leftMargin = 880;
                layoutParams.bottomMargin = 350;
                rotationAngle = 0f; // 设置旋转角度为45度
                break;
            case "6-2":
                layoutParams.leftMargin = 900;
                layoutParams.bottomMargin = 360;
                rotationAngle = 90f; // 设置旋转角度为45度
                break;
            case "6-3":
                layoutParams.leftMargin = 880;
                layoutParams.bottomMargin = 360;
                rotationAngle = 180f; // 设置旋转角度为45度
                break;
            case "7-1":
                layoutParams.leftMargin = 880;
                layoutParams.bottomMargin = 310;
                rotationAngle = 0f; // 设置旋转角度为45度
                break;
            case "8-1":
                layoutParams.leftMargin = 880;
                layoutParams.bottomMargin = 360;
                rotationAngle = 270f; // 设置旋转角度为45度
                break;
            case "8-2":
                layoutParams.leftMargin = 890;
                layoutParams.bottomMargin = 360;
                rotationAngle = 180f; // 设置旋转角度为45度
                break;
            case "8-3":
                layoutParams.leftMargin = 990;
                layoutParams.bottomMargin = 360;
                rotationAngle = 90f; // 设置旋转角度为45度
                break;
            default:
                layoutParams.leftMargin = 0;
                layoutParams.bottomMargin = 0;
                rotationAngle = 0f; // 设置旋转角度为45度
                break;
        }


        Log.d("Tag", String.valueOf(layoutParams.leftMargin) + "?LEFT11111111111111?");  // 打印调试级别的日志消息
        Log.d("Tag", String.valueOf(layoutParams.bottomMargin) + "?TOP11111111111111?");  // 打印调试级别的日志消息
        vector.setVisibility(View.VISIBLE);
        vector.setLayoutParams(layoutParams);
        vector.setRotation(rotationAngle);
//
//        inputImageBuffer.load(resizedBitmap);
//        TensorImage inputTensorImage = imageProcessor.process(inputImageBuffer);
//
//        tfliteInterpreter.run(inputTensorImage.getBuffer(), outputBuffer.getBuffer().rewind());
//
//        float[] results = outputBuffer.getFloatArray();
//        String inferenceResult = processInferenceResult(results);
//        resultTextView.setText("推論結果: " + inferenceResult);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabels() {
        List<String> labels = new ArrayList<>();
        AssetManager assetManager = getAssets();

        try {
            InputStream inputStream = assetManager.open("labels.txt");
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return labels;
    }

    private String processInferenceResult(float[] results) {
        int maxIndex = 0;
        float maxProb = results[0];

        for (int i = 1; i < results.length; i++) {
            if (results[i] > maxProb) {
                maxIndex = i;
                maxProb = results[i];
            }
        }

        return classLabels.get(maxIndex);
    }
}
