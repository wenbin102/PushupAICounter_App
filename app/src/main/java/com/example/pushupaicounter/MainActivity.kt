package com.example.pushupaicounter

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PointF
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.pushupaicounter.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.mlkit.vision.MlKitAnalyzer
import androidx.camera.view.CameraController.COORDINATE_SYSTEM_VIEW_REFERENCED
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import org.apache.commons.math3.complex.Complex
import org.apache.commons.math3.transform.DftNormalization
import org.apache.commons.math3.transform.FastFourierTransformer
import org.apache.commons.math3.transform.TransformType
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.min


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var poseDetector: PoseDetector
    private lateinit var cameraExecutor: ExecutorService
//    private lateinit var poseDetectorResults: Pose
    private lateinit var canvas: Canvas
    private lateinit var pushup_count: TextView
    private lateinit var counting_button: Button
    lateinit var peaks: List<Int>
    private var pushUpCount = 0
    private var start_timer = false
    private var start_time: Long = 0
    private var current_time: Long = 0


    var paint = Paint()

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && it.value == false)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT).show()
            } else {
                startCamera()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        var cameraController = LifecycleCameraController(baseContext)
        val previewView: PreviewView = viewBinding.viewFinder
        pushup_count = findViewById<TextView>(R.id.pushup_count)
        counting_button = findViewById<Button>(R.id.counting_button)
        var elbow_l = 0.0
        var elbow_r = 0.0
        var shoulder_l = 0.0
        var shoulder_r = 0.0
        var elbow_angle = 0.0
        var shoulder_angle = 0.0
        val threshold = 0.4
        var elsh: DoubleArray = doubleArrayOf() //elbow + shoulder angle
        val transformer = FastFourierTransformer(DftNormalization.STANDARD)
        var begin_time: Long = 0
        var ending_time: Long = 0

        counting_button.setOnClickListener({start_timer = true})


        val options = PoseDetectorOptions.Builder()
            .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
            .build()

        poseDetector = PoseDetection.getClient(options)

        cameraController.setImageAnalysisAnalyzer(
            ContextCompat.getMainExecutor(this),
            MlKitAnalyzer(
                listOf(poseDetector),
                COORDINATE_SYSTEM_VIEW_REFERENCED,
                ContextCompat.getMainExecutor(this)
            ) { result: MlKitAnalyzer.Result? ->
                val poseDetectorResults = result?.getValue(poseDetector)
                begin_time = System.currentTimeMillis()
                if (poseDetectorResults == null
                ) {
                    previewView.setOnTouchListener { _, _ -> false } //no-op //to ignore touch event

                    return@MlKitAnalyzer
                }

                val leftShoulder = poseDetectorResults.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)?.position
                val rightShoulder = poseDetectorResults.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)?.position
                val leftElbow = poseDetectorResults.getPoseLandmark(PoseLandmark.LEFT_ELBOW)?.position
                val rightElbow = poseDetectorResults.getPoseLandmark(PoseLandmark.RIGHT_ELBOW)?.position
                val leftWrist = poseDetectorResults.getPoseLandmark(PoseLandmark.LEFT_WRIST)?.position
                val rightWrist = poseDetectorResults.getPoseLandmark(PoseLandmark.RIGHT_WRIST)?.position

                val leftHip = poseDetectorResults.getPoseLandmark(PoseLandmark.LEFT_HIP)?.position
                val rightHip = poseDetectorResults.getPoseLandmark(PoseLandmark.RIGHT_HIP)?.position

                if (
                    leftShoulder != null && rightShoulder != null && leftElbow != null && rightElbow != null && leftWrist != null && rightWrist != null && leftHip != null && rightHip != null
                ){
                    elbow_l = findAngle(poseDetectorResults, leftShoulder, leftElbow, leftWrist)
                    elbow_r = findAngle(poseDetectorResults, rightShoulder, rightElbow, rightWrist)
                    shoulder_l = findAngle(poseDetectorResults, leftElbow, leftShoulder, leftHip)
                    shoulder_r = findAngle(poseDetectorResults, rightElbow, rightShoulder, rightHip)

                    shoulder_angle = min(shoulder_l, shoulder_r)

                    elbow_angle = min(elbow_l, elbow_r)

                    elsh += (elbow_angle + shoulder_angle)
                }

                if (isPowerOfTwo(elsh.size)) {
                    val y = transformer.transform(elsh, TransformType.FORWARD)

                    val n = elsh.size
                    ending_time = System.currentTimeMillis()
                    val sampleRate = (ending_time - begin_time).toDouble() //1/5   //change the sample rate
                    val freq = DoubleArray(n) { it / (n * sampleRate) }

                    for (i in 0 until n) {
                        if (Math.abs(freq[i]) > threshold) {
                            y[i] = Complex(0.0, 0.0)
                        }
                    }

                    // Calculate the inverse FFT (Inverse Fast Fourier Transform)
                    var filtered = transformer.transform(y, TransformType.INVERSE)

                    val filteredReal = filtered.map { it.real }.toDoubleArray() //Complex numbers to real numbers

                    peaks = findPeaks(filteredReal) //the peaks which stands for one pushup each peak

                    pushUpCount = peaks.size
                }

                if (start_timer == true){
                    pushup_count.text = "0"
                    start_timer = false
                    start_time = System.currentTimeMillis()
                }

                current_time = System.currentTimeMillis()

                if (current_time - start_time < 20000){
                    pushup_count.text = pushUpCount.toString()
                }


//                val nosePosition = poseDetectorResults?.getPoseLandmark(PoseLandmark.NOSE)?.position
//                var x :Float = 0.0f
//                var y: Float = 0.0f
//                if (nosePosition != null) {
//                    // Use landmarkPosition safely
//                    x = nosePosition.x
//                    y = nosePosition.y
//                }
//
//                var puc = x.toString()
//
//                pushup_count.text = puc


                // Set up the listeners for counter buttons
                // viewBinding.countingButton.setOnClickListener { startCount() }

                previewView.overlay.clear()
            }
        )

        cameraController.bindToLifecycle(this)
        previewView.controller = cameraController
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()

    }

    private fun findAngle(pose: Pose, a: PointF, b: PointF, c: PointF): Double {
        var x1 = a.x
        var x2 = b.x
        var x3 = c.x
        var y1 = a.y
        var y2 = b.y
        var y3 = c.y

        var angle_rad = (atan2((y3 - y2).toDouble(), (x3 - x2).toDouble()) - atan2(
            (y1 - y2).toDouble(),
            (x1 - x2).toDouble()
        ))
        var angle = angle_rad * 180 / PI

        if (angle < 0) {
            angle += 360
            if (angle > 180) {
                angle = 360 - angle
            }
        } else if (angle > 180) {
            angle = 360 - angle
        }

        return angle

    }

    fun findPeaks(signal: DoubleArray): List<Int> {
        val peaks = mutableListOf<Int>()

        for (i in 1 until signal.size - 1) {
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
                peaks.add(i)
            }
        }

        return peaks
    }

    fun isPowerOfTwo(number: Int): Boolean {
        return number > 0 && (number and (number - 1)) == 0
    }

    companion object {
        private const val TAG = "PushUp AI Counter App"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }
}