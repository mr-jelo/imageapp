package com.example.mlkitimageapp

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.min
import kotlin.math.exp

class MainActivity : AppCompatActivity() {
    private lateinit var selectImageButton: Button
    private lateinit var selectedImageView: ImageView
    private lateinit var labelResults: TextView
    private val REQUEST_IMAGE_PICK = 100
    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        selectImageButton = findViewById(R.id.selectImageButton)
        selectedImageView = findViewById(R.id.selectedImageView)
        labelResults = findViewById(R.id.labelResults)

        try {
            val modelFile = FileUtil.loadMappedFile(this, "model.tflite")
            val options = Interpreter.Options()
            tflite = Interpreter(modelFile, options)
            labels = FileUtil.loadLabels(this, "labels.txt")
        } catch (e: Exception) {
            Log.e("TFLite", "Failed to load model or labels.", e)
            Toast.makeText(this, "Failed to load model.", Toast.LENGTH_SHORT).show()
        }

        selectImageButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, REQUEST_IMAGE_PICK)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_IMAGE_PICK && resultCode == Activity.RESULT_OK) {
            val imageUri: Uri? = data?.data
            imageUri?.let {
                selectedImageView.setImageURI(it)
                try {
                    val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
                    runTFLiteInference(bitmap)
                } catch (e: Exception) {
                    labelResults.text = "Failed to load image: ${e.message}"
                    Log.e("ImageError", "Bitmap load failed", e)
                }
            }
        }
    }

    private fun runTFLiteInference(bitmap: Bitmap) {
        val inputShape = tflite.getInputTensor(0).shape()
        val inputHeight = inputShape[1]
        val inputWidth = inputShape[2]

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        val outputShape = tflite.getOutputTensor(0).shape()
        val outputTensor = TensorBuffer.createFixedSize(outputShape, tflite.getOutputTensor(0).dataType())

        tflite.run(processedImage.buffer, outputTensor.buffer)
        val outputRawArray = outputTensor.floatArray

        val probabilities = softmax (outputRawArray)

        val labeledProbability = mutableMapOf<String, Float>()
        for (i in labels.indices) {
            if (i < probabilities.size) {
                labeledProbability[labels[i]] = probabilities[i]
            }
        }

        val topResults = getTopKResults(labeledProbability, 10)

        val results = topResults.joinToString("\n") { (label, probability) ->
            "${label}: ${String.format("%.2f", probability * 100)}%"
        }

        labelResults.text = results
        Toast.makeText(this, "Recognition complete", Toast.LENGTH_SHORT).show()
    }

    private fun getTopKResults(labeledProbability: Map<String, Float>, topK: Int): List<Pair<String, Float>> {
        val sortedResults = labeledProbability.toList().sortedByDescending { (_, value) -> value }
        return sortedResults.take(min(topK, sortedResults.size))
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0.0f
        val exps = logits.map { exp(it - maxLogit) }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }
}
