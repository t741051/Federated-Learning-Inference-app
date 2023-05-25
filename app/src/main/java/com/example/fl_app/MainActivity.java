package com.example.fl_app;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

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
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private static final int INPUT_IMAGE_SIZE = 75;
    private static final int NUM_CLASSES = 17;

    private ImageView imageView;
    private TextView resultTextView;
    private Button inferenceButton;

    private Interpreter tfliteInterpreter;
    private ImageProcessor imageProcessor;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputBuffer;
    private List<String> classLabels;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        resultTextView = findViewById(R.id.resultTextView);
        inferenceButton = findViewById(R.id.inferenceButton);

        inferenceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openImagePicker();
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
    }

    private void performInference(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
//
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
