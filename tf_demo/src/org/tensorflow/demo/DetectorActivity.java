/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.util.Log;
import android.view.Display;
import android.view.Surface;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Vector;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.demo.R; // Explicit import needed for internal Google builds.
import org.tensorflow.demo.tracking.Tuple;
import org.tensorflow.demo.tracking.Triplet;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener, AdapterView.OnItemSelectedListener {
  private static final Logger LOGGER = new Logger();

  // Pete added this variable to make it easier to change thresholds and class names
  private static String CROP = "faw";
  private static int class_array = 0;

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 224;
  private static final int MB_IMAGE_MEAN = 128;
  private static final float MB_IMAGE_STD = 128;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
        "file:///android_asset/faw_detect_52.3.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/faw_label_list.txt";
  private static int CLASS_1_COLOR = Color.RED; // CBSD
  private static int CLASS_2_COLOR = Color.MAGENTA; // CMD
  private static int CLASS_3_COLOR = Color.GREEN; // CGM
  private static int CLASS_4_COLOR = Color.BLUE; // CRM
  private static int CLASS_5_COLOR = Color.YELLOW; // CBLS
  private static int CLASS_6_COLOR = Color.WHITE; // Healthy
  private static int CLASS_7_COLOR = Color.BLACK; // CND

  private static String CLASS_1_NAME = "CBSD";
  private static String CLASS_2_NAME = "CMD";
  private static String CLASS_3_NAME = "CGM";
  private static String CLASS_4_NAME = "CRM";
  private static String CLASS_5_NAME = "CBLS";
  private static String CLASS_6_NAME = "Healthy";
  private static String CLASS_7_NAME = "CND";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
  private static final int YOLO_INPUT_SIZE = 416;
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";
  private static final int YOLO_BLOCK_SIZE = 32;



  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
  // or YOLO.
  private enum DetectorMode {
    TF_OD_API, MULTIBOX, YOLO;
  }
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
  private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;


  private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

  public String groundTruthClass;

  // All counts below have inclusive upper bounds only
  private int interval0 = 0; // Counts for 0 to 10 % confidence
  private int interval1 = 0; // Counts for 10 to 20 % confidence
  private int interval2 = 0; // Counts for 20 to 30 % confidence
  private int interval3 = 0; // Counts for 30 to 40 % confidence
  private int interval4 = 0; // Counts for 40 to 50 % confidence
  private int interval5 = 0; // Counts for 50 to 60 % confidence
  private int interval6 = 0; // Counts for 60 to 70 % confidence
  private int interval7 = 0; // Counts for 70 to 80 % confidence
  private int interval8 = 0; // Counts for 80 to 90 % confidence
  private int interval9 = 0; // Counts for 90 to 100 % confidence

  // All counts below have inclusive upper bounds only
  private int cinterval0 = 0; // Correct counts for 0 to 10 % confidence
  private int cinterval1 = 0; // Correct counts for 10 to 20 % confidence
  private int cinterval2 = 0; // Correct counts for 20 to 30 % confidence
  private int cinterval3 = 0; // Correct counts for 30 to 40 % confidence
  private int cinterval4 = 0; // Correct counts for 40 to 50 % confidence
  private int cinterval5 = 0; // Correct counts for 50 to 60 % confidence
  private int cinterval6 = 0; // Correct counts for 60 to 70 % confidence
  private int cinterval7 = 0; // Correct counts for 70 to 80 % confidence
  private int cinterval8 = 0; // Correct counts for 80 to 90 % confidence
  private int cinterval9 = 0; // Correct cunts for 90 to 100 % confidence


  // Percentages for confidence counters
  private float per0 = 0;
  private float per1 = 0;
  private float per2 = 0;
  private float per3 = 0;
  private float per4 = 0;
  private float per5 = 0;
  private float per6 = 0;
  private float per7 = 0;
  private float per8 = 0;
  private float per9 = 0;

  // Adding button functionality for start/stop detection tracking
  boolean record = false;

  private ArrayList<Float> calArray = new ArrayList<Float>(); // Array of all confidence values
  private ArrayList<Triplet> lastSet = new ArrayList<Triplet>(); // Latest set of detections

  protected void onCreate(Bundle calTest) {


    super.onCreate(calTest);
    setContentView(R.layout.activity_camera);

    final ImageView circleView = (ImageView) findViewById(R.id.cView);
    circleView.setVisibility(View.GONE);

    final Button calButton = (Button) findViewById(R.id.calButton);
    calButton.setOnClickListener(new View.OnClickListener() {
      public void onClick(View v) {
        record = !record;
        if (record) {
          circleView.setVisibility(View.VISIBLE);
        } else {
          circleView.setVisibility(View.GONE);
        }
      }
    });

    // Clear array button functionality
    final Button clearButton = (Button) findViewById(R.id.clearButton);
    clearButton.setOnClickListener(new View.OnClickListener() {
      public void onClick(View v) {
        calArray.clear();
        interval0 = 0;
        interval1 = 0;
        interval2 = 0;
        interval3 = 0;
        interval4 = 0;
        interval5 = 0;
        interval6 = 0;
        interval7 = 0;
        interval8 = 0;
        interval9 = 0;

        cinterval0 = 0;
        cinterval1 = 0;
        cinterval2 = 0;
        cinterval3 = 0;
        cinterval4 = 0;
        cinterval5 = 0;
        cinterval6 = 0;
        cinterval7 = 0;
        cinterval8 = 0;
        cinterval9 = 0;

        per0 = 0;
        per1 = 0;
        per2 = 0;
        per3 = 0;
        per4 = 0;
        per5 = 0;
        per6 = 0;
        per7 = 0;
        per8 = 0;
        per9 = 0;
      }
    });

    if (CROP.equals("cassava")) {
        class_array = R.array.cassava_array;
    } else if (CROP.equals("faw")) {
        class_array = R.array.faw_array;
    }

    Spinner spinner = (Spinner) findViewById(R.id.spinner1);
    // Create an ArrayAdapter using the string array and a default spinner layout
    ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
            class_array, android.R.layout.simple_spinner_item);
    // Specify the layout to use when the list of choices appears
    adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
    // Apply the adapter to the spinner
    spinner.setAdapter(adapter);

    spinner.setOnItemSelectedListener(this);

  }

  @Override
  public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {

    groundTruthClass = parent.getSelectedItem().toString();

  }

  @Override
  public void onNothingSelected(AdapterView<?> parent) {
    // TODO Auto-generated method stub
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {

    if (CROP.equals("cassava")) {
      CLASS_1_NAME = "CBSD";
      CLASS_2_NAME = "CMD";
      CLASS_3_NAME = "CGM";
      CLASS_4_NAME = "CRM";
      CLASS_5_NAME = "CBLS";
      CLASS_6_NAME = "Healthy";
      CLASS_7_NAME = "CND";

      CLASS_1_COLOR = Color.RED; // CBSD
      CLASS_2_COLOR = Color.MAGENTA; // CMD
      CLASS_3_COLOR = Color.GREEN; // CGM
      CLASS_4_COLOR = Color.BLUE; // CRM
      CLASS_5_COLOR = Color.WHITE; // HEALTHY
      CLASS_6_COLOR = Color.YELLOW; // CBLS
      CLASS_7_COLOR = Color.BLACK; // CND

    } else if (CROP.equals("faw")) {
      CLASS_1_NAME = "FAWLeaf";
      CLASS_2_NAME = "FAWFrass";

      CLASS_1_COLOR = Color.RED; // FAWLeaf
      CLASS_2_COLOR = Color.MAGENTA; // FAWFrass
    } else if (CROP.equals("wheat")) {
      CLASS_1_NAME = "WheatStemRustStem";
      CLASS_2_NAME = "WheatStemRustLeaf";
      CLASS_3_NAME = "WheatHL";
      CLASS_4_NAME = "WheatHS";
      CLASS_5_NAME = "WheatStripeRustLeaf";

      CLASS_1_COLOR = Color.RED; // WheatStemRustStem
      CLASS_2_COLOR = Color.MAGENTA; // WheatStemRustLeaf
      CLASS_3_COLOR = Color.GREEN; // WheatHL
      CLASS_4_COLOR = Color.BLUE; // WheatHS
      CLASS_5_COLOR = Color.WHITE; // WheatStripeRustLeaf

    }

    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;
    if (MODE == DetectorMode.YOLO) {
      detector =
          TensorFlowYoloDetector.create(
              getAssets(),
              YOLO_MODEL_FILE,
              YOLO_INPUT_SIZE,
              YOLO_INPUT_NAME,
              YOLO_OUTPUT_NAMES,
              YOLO_BLOCK_SIZE);
      cropSize = YOLO_INPUT_SIZE;
    } else if (MODE == DetectorMode.MULTIBOX) {
      detector =
          TensorFlowMultiBoxDetector.create(
              getAssets(),
              MB_MODEL_FILE,
              MB_LOCATION_FILE,
              MB_IMAGE_MEAN,
              MB_IMAGE_STD,
              MB_INPUT_NAME,
              MB_OUTPUT_LOCATIONS_NAME,
              MB_OUTPUT_SCORES_NAME);
      cropSize = MB_INPUT_SIZE;
    } else {
      try {
        detector = TensorFlowObjectDetectionAPIModel.create(
            getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        LOGGER.e("Exception initializing classifier!", e);
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    final Lock lock = new ReentrantLock();
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            boolean triggerCount;
            ArrayList<Triplet> conTrips = tracker.draw(canvas, groundTruthClass); // look into draw callback api

            // Comparing detection set against last set of detections to determine whether to count new set or not
            // Detections that are repeated from the last set will not be double counted
            if (compare(conTrips, lastSet)) {
              triggerCount = false;
            } else {
              triggerCount = true;
            }


            if (isDebug() && triggerCount) {
              //tracker.drawDebug(canvas);
              if (record) {
                // Looping through detection set and incrementing confidence counters accordingly
                for (Triplet<Float, Boolean, Tuple> conTrip : conTrips) {
                  Float conVal = conTrip.getFirst();
                  Boolean conCorrect = conTrip.getSecond();
                  if (conVal != 0) {
                      calArray.add(conVal);
                      if ((conVal > 0.0) && (conVal <= 0.1)) {
                        interval0 += 1;
                        cinterval0 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.1) && (conVal <= 0.2)) {
                        interval1 += 1;
                        cinterval1 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.2) && (conVal <= 0.3)) {
                        interval2 += 1;
                        cinterval2 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.3) && (conVal <= 0.4)) {
                        interval3 += 1;
                        cinterval3 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.4) && (conVal <= 0.5)) {
                        interval4 += 1;
                        cinterval4 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.5) && (conVal <= 0.6)) {
                        interval5 += 1;
                        cinterval5 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.6) && (conVal <= 0.7)) {
                        interval6 += 1;
                        cinterval6 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.7) && (conVal <= 0.8)) {
                        interval7 += 1;
                        cinterval7 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.8) && (conVal <= 0.9)) {
                        interval8 += 1;
                        cinterval8 += (conCorrect == true) ? 1 : 0;
                      } else if ((conVal > 0.9) && (conVal <= 1.0)) {
                        interval9 += 1;
                        cinterval9 += (conCorrect == true) ? 1 : 0;
                      }
                    }
                  }

                  per0 = (float) cinterval0 / interval0;
                  per1 = (float) cinterval1 / interval1;
                  per2 = (float) cinterval2 / interval2;
                  per3 = (float) cinterval3 / interval3;
                  per4 = (float) cinterval4 / interval4;
                  per5 = (float) cinterval5 / interval5;
                  per6 = (float) cinterval6 / interval6;
                  per7 = (float) cinterval7 / interval7;
                  per8 = (float) cinterval8 / interval8;
                  per9 = (float) cinterval9 / interval9;
                }
              }
              lastSet = conTrips;
            }
        });

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }
            // Color 'tint' for volume down stats overlay
            final int backgroundColor = Color.argb(0, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            //final Matrix matrix = new Matrix();
            //final float scaleFactor = 2;
            //matrix.postScale(scaleFactor, scaleFactor);
            //matrix.postTranslate(
            //    canvas.getWidth() - copy.getWidth() * scaleFactor,
            //    canvas.getHeight() - copy.getHeight() * scaleFactor);
            //canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();

            if (record) {
              lines.add("RECORDING ON");
            } else {
              lines.add("RECORDING OFF");
            }

            for (Float confidence: calArray) {
              lines.add(String.valueOf(confidence));
            }

            // Displaying confidences on screen
            lines.add("0 to 10: " + String.valueOf(interval0) + " " + String.valueOf(cinterval0) + " " + Float.toString(per0));
            lines.add("10 to 20: " + String.valueOf(interval1) + " " + String.valueOf(cinterval1) + " " + Float.toString(per1));
            lines.add("20 to 30: " + String.valueOf(interval2) + " " + String.valueOf(cinterval2) + " " + Float.toString(per2));
            lines.add("30 to 40: " + String.valueOf(interval3) + " " + String.valueOf(cinterval3) + " " + Float.toString(per3));
            lines.add("40 to 50: " + String.valueOf(interval4) + " " + String.valueOf(cinterval4) + " " + Float.toString(per4));
            lines.add("50 to 60: " + String.valueOf(interval5) + " " + String.valueOf(cinterval5) + " " + Float.toString(per5));
            lines.add("60 to 70: " + String.valueOf(interval6) + " " + String.valueOf(cinterval6) + " " + Float.toString(per6));
            lines.add("70 to 80: " + String.valueOf(interval7) + " " + String.valueOf(cinterval7) + " " + Float.toString(per7));
            lines.add("80 to 90: " + String.valueOf(interval8) + " " + String.valueOf(cinterval8) + " " + Float.toString(per8));
            lines.add("90 to 100: " + String.valueOf(interval9) + " " + String.valueOf(cinterval9) + " " + Float.toString(per9));


            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);

          }
        });
  }

  // Deep comparison of conPairs and lastSet contents to check for repeated detections
  private Boolean compare(ArrayList<Triplet> conTrips, ArrayList<Triplet> lastSet) {

    if (conTrips.size() != lastSet.size()) {
      return false;
    }

    for (int x = 0; x < conTrips.size(); x++) {
      Tuple<Float, Float> conTripLocation = (Tuple<Float, Float>) conTrips.get(x).getThird();
      Tuple<Float, Float> lastSetLocation = (Tuple<Float, Float>) lastSet.get(x).getThird();
      if (Math.floor(((float) conTrips.get(x).getFirst()) * 1000) != Math.floor(((float) lastSet.get(x).getFirst()) * 1000)) {
        return false;
      } else if (compareDistance(conTripLocation.getFirst(), conTripLocation.getSecond(), lastSetLocation.getFirst(), lastSetLocation.getSecond())) {
        return false;
      }

    }

    return true;
  }

  private Boolean compareDistance(float conTripX, float conTripY, float lastSetX, float lastSetY) {
    if (Math.sqrt((lastSetY - conTripY) * (lastSetY - conTripY) + (lastSetX - conTripX) * (lastSetX - conTripX)) > 4) {
      return true;
    } else {
      return false;
    }
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
              case MULTIBOX:
                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                break;
              case YOLO:
                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final String className = result.getId();
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                if (className.equals(CLASS_1_NAME)) {
                  paint.setColor(CLASS_1_COLOR);
                } else if (className.equals(CLASS_2_NAME)) {
                  paint.setColor(CLASS_2_COLOR);
                } else if (className.equals(CLASS_3_NAME)) {
                  paint.setColor(CLASS_3_COLOR);
                } else if (className.equals(CLASS_4_NAME)) {
                  paint.setColor(CLASS_4_COLOR);
                } else if (className.equals(CLASS_5_NAME)) {
                  paint.setColor(CLASS_5_COLOR);
                } else if (className.equals(CLASS_6_NAME)) {
                  paint.setColor(CLASS_6_COLOR);
                } else if (className.equals(CLASS_7_NAME)) {
                  paint.setColor(CLASS_7_COLOR);
                }
                canvas.drawRect(location, paint);
                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();

            requestRender();
            computingDetection = false;
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }
}
