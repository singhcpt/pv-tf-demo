/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.demo.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Log;
import android.util.TypedValue;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.tensorflow.demo.Classifier.Recognition;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.Tuple;



/**
 * A tracker wrapping ObjectTracker that also handles non-max suppression and matching existing
 * objects to new detections.
 */
public class MultiBoxTracker {
  private final Logger logger = new Logger();
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

  private static float CLASS_1_THRESHOLD = 0.3f;    // FAW Leaf Damage
  private static float CLASS_2_THRESHOLD = 0.3f;   // Frass
  private static float CLASS_3_THRESHOLD = 0.3f;   // Wet Frass
  private static float CLASS_4_THRESHOLD = 0.3f;   // CBSD
  private static float CLASS_5_THRESHOLD = 0.3f;   // CMD
  private static float CLASS_6_THRESHOLD = 0.3f;   // CGM
  private static float CLASS_7_THRESHOLD = 0.3f;   // CRM

  private static String CROP = "faw";

  private static final float MAX_SIZE = 400.0f;
  private static final float MIN_SIZE = 10.0f;
  // Allow replacement of the tracked box with new results if
  // correlation has dropped below this level.
  private static final float MARGINAL_CORRELATION = 0.75f;
  // Consider object to be lost if correlation falls below this threshold.
  private static final float MIN_CORRELATION = 0.1f;
  // Maximum percentage of a box that can be overlapped by another box at detection time. Otherwise
  // the lower scored box (new or old) will be removed.
  private static final float MAX_OVERLAP = 0.8f;
  private static float DISPLAY_THRESHOLD = 0.2f;
  private static final float DETECTION_THRESHOLD = 0.1f;
  private static final float TEXT_SIZE_DISPLAY = 11;
  private static final float DETECTION_BOX_LINE_THICKNESS = 4.5f;

  private static final int DASH_LENGTH = 30;
  private static final int DASH_GAP = 10;

  private static final int MIN_BOXES = 1;
  private static int numTracked = 0;
  final Lock lock = new ReentrantLock();


    private static final LinkedList<String> shownDetections = new LinkedList<String>();

  private static final int[] COLORS = {
          Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.WHITE,
          Color.parseColor("#55FF55"), Color.parseColor("#FFA500"), Color.parseColor("#FF8888"),
          Color.parseColor("#AAAAFF"), Color.parseColor("#FFFFAA"), Color.parseColor("#55AAAA"),
          Color.parseColor("#AA33AA"), Color.parseColor("#0D0068")
  };

  private final Queue<Integer> availableColors = new LinkedList<Integer>();

  public ObjectTracker objectTracker;

  final List<Triplet<String, Float, RectF>> screenRects = new LinkedList<Triplet<String, Float, RectF>>();

  private static class TrackedRecognition {
    ObjectTracker.TrackedObject trackedObject;
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }

  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();

  private final Paint boxPaint = new Paint();

  private final float textSizePx;
  private final BorderedText borderedText;

  private Matrix frameToCanvasMatrix;

  private int frameWidth;
  private int frameHeight;

  private int sensorOrientation;
  private Context context;


  public MultiBoxTracker(final Context context) {

      if (CROP.equals("cassava")) {
          CLASS_1_NAME = "CBSD";
          CLASS_2_NAME = "CMD";
          CLASS_3_NAME = "CGM";
          CLASS_4_NAME = "CRM";
          CLASS_5_NAME = "CBLS";
          CLASS_6_NAME = "Healthy";
          CLASS_7_NAME = "CND";

          CLASS_1_THRESHOLD = 0.5f;
          CLASS_2_THRESHOLD = 0.5f;
          CLASS_3_THRESHOLD = 0.5f;
          CLASS_4_THRESHOLD = 0.5f;
          CLASS_5_THRESHOLD = 0.5f;
          CLASS_6_THRESHOLD = 0.5f;
          CLASS_7_THRESHOLD = 0.5f;

          CLASS_1_COLOR = Color.RED; // CBSD
          CLASS_2_COLOR = Color.MAGENTA; // CMD
          CLASS_3_COLOR = Color.GREEN; // CGM
          CLASS_4_COLOR = Color.BLUE; // CRM
          CLASS_5_COLOR = Color.YELLOW; // CBLS
          CLASS_6_COLOR = Color.WHITE; // Healthy
          CLASS_7_COLOR = Color.BLACK; // CND

      } else if (CROP.equals("faw")) {
          CLASS_1_NAME = "FAWLeaf";
          CLASS_2_NAME = "FAWFrass";

          CLASS_1_THRESHOLD = 0.05f;
          CLASS_2_THRESHOLD = 0.05f;

          CLASS_1_COLOR = Color.RED; // FAWLeaf
          CLASS_2_COLOR = Color.MAGENTA; // FAWFrass
      } else if (CROP.equals("wheat")) {
          CLASS_1_NAME = "WheatStemRustStem";
          CLASS_2_NAME = "WheatStemRustLeaf";
          CLASS_3_NAME = "WheatHL";
          CLASS_4_NAME = "WheatHS";
          CLASS_5_NAME = "WheatStripeRustLeaf";

          CLASS_1_THRESHOLD = 0.3f;
          CLASS_2_THRESHOLD = 0.3f;
          CLASS_3_THRESHOLD = 0.3f;
          CLASS_4_THRESHOLD = 0.3f;
          CLASS_5_THRESHOLD = 0.3f;

          CLASS_1_COLOR = Color.RED; // WheatStemRustStem
          CLASS_2_COLOR = Color.MAGENTA; // WheatStemRustLeaf
          CLASS_3_COLOR = Color.GREEN; // WheatHL
          CLASS_4_COLOR = Color.BLUE; // WheatHS
          CLASS_5_COLOR = Color.WHITE; // WheatStripeRustLeaf

      }

      shownDetections.add(CLASS_1_NAME);
      shownDetections.add(CLASS_2_NAME);
      shownDetections.add(CLASS_3_NAME);
      shownDetections.add(CLASS_6_NAME);


      this.context = context;
      for (final int color : COLORS) {
          availableColors.add(color);
      }
      boxPaint.setColor(Color.RED);
      boxPaint.setStyle(Style.STROKE);
      boxPaint.setStrokeWidth(12.0f);
      boxPaint.setStrokeCap(Cap.ROUND);
      boxPaint.setStrokeJoin(Join.ROUND);
      boxPaint.setStrokeMiter(100);
      textSizePx =
              TypedValue.applyDimension(
                      TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DISPLAY, context.getResources().getDisplayMetrics());
      borderedText = new BorderedText(textSizePx);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    Log.d("myInfoTag","IN DRAW DEBUG");
    final Paint boxPaint = new Paint();
    boxPaint.setAlpha(10);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(DETECTION_BOX_LINE_THICKNESS);
    boxPaint.setPathEffect(new DashPathEffect(new float[] {DASH_LENGTH, DASH_GAP}, 0));

    for (final Triplet<String, Float, RectF> detection : screenRects) {
        final RectF rect = detection.getThird();
        final String className = detection.getFirst();
        final Float confidence = detection.getSecond();

        final String labelString =
              !TextUtils.isEmpty(className)
                      ? String.format("%s %.0f%%", className, confidence*100)
                      : String.format("%.0f%%", confidence);
        if (className.equals(CLASS_1_NAME)) {
            DISPLAY_THRESHOLD = CLASS_1_THRESHOLD;
            boxPaint.setColor(CLASS_1_COLOR);
        } else if (className.equals(CLASS_2_NAME)) {
            DISPLAY_THRESHOLD = CLASS_2_THRESHOLD;
            boxPaint.setColor(CLASS_2_COLOR);
        } else if (className.equals(CLASS_3_NAME)) {
            DISPLAY_THRESHOLD = CLASS_3_THRESHOLD;
            boxPaint.setColor(CLASS_3_COLOR);
        } else if (className.equals(CLASS_4_NAME)) {
            DISPLAY_THRESHOLD = CLASS_4_THRESHOLD;
            boxPaint.setColor(CLASS_4_COLOR);
        } else if (className.equals(CLASS_5_NAME)) {
            DISPLAY_THRESHOLD = CLASS_5_THRESHOLD;
            boxPaint.setColor(CLASS_5_COLOR);
        } else if (className.equals(CLASS_6_NAME)) {
            DISPLAY_THRESHOLD = CLASS_6_THRESHOLD;
            boxPaint.setColor(CLASS_6_COLOR);
        } else if (className.equals(CLASS_7_NAME)) {
            DISPLAY_THRESHOLD = CLASS_7_THRESHOLD;
            boxPaint.setColor(CLASS_7_COLOR);
        }
        if (confidence>= DETECTION_THRESHOLD && confidence < DISPLAY_THRESHOLD) {
            canvas.drawRect(rect, boxPaint);
            borderedText.drawText(canvas, rect.left, rect.top, labelString);
        }

    }

    // Draw correlations.
    for (final TrackedRecognition recognition : trackedObjects) {
      final ObjectTracker.TrackedObject trackedObject = recognition.trackedObject;

      final RectF trackedPos = trackedObject.getTrackedPositionInPreviewFrame();

      if (getFrameToCanvasMatrix().mapRect(trackedPos)) {
        final String labelString = String.format("%.2f", trackedObject.getCurrentCorrelation());
        borderedText.drawText(canvas, trackedPos.right, trackedPos.bottom, labelString);
      }
    }

    final Matrix matrix = getFrameToCanvasMatrix();
    objectTracker.drawDebug(canvas, matrix);
  }

  public synchronized void trackResults(
      final List<Recognition> results, final byte[] frame, final long timestamp) {
    //logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(timestamp, results, frame);
  }

  public synchronized ArrayList<Triplet> draw(final Canvas canvas, final String groundTruthClass) {
    ArrayList<Triplet> conVals = new ArrayList<Triplet>();
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
        Math.min(canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                 canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false);
    for (final TrackedRecognition recognition : trackedObjects) {
        final String className = recognition.title;
        final Float confidence = recognition.detectionConfidence;

        boolean correct = false;

        final RectF trackedPos =
          (objectTracker != null)
              ? recognition.trackedObject.getTrackedPositionInPreviewFrame()
              : new RectF(recognition.location);
        final float width = recognition.location.width();
        final float height = recognition.location.height();

        if (confidence >= 0.5) {
            if (className.equals(groundTruthClass)) { // put parameter
                correct = true;
            }
            Triplet conTrip = new Triplet<Float, Boolean, Tuple>(confidence, correct, new Tuple<Float, Float>(recognition.location.centerX(), recognition.location.centerY()));
            if (!(conVals.contains(conTrip))) {
                conVals.add(conTrip);
            }
        }

      getFrameToCanvasMatrix().mapRect(trackedPos);
      final float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;

      // Now lets get ready to draw the box and cooresponding label.
      final String labelString =
          !TextUtils.isEmpty(className)
              ? String.format("%s %.5f%%", className, confidence)
              : String.format("%.0f%%", confidence*100);

      //  Determine which color to draw the box based on the class name
        if (className.equals(CLASS_1_NAME)) {
            DISPLAY_THRESHOLD = CLASS_1_THRESHOLD;
            boxPaint.setColor(CLASS_1_COLOR);
        } else if (className.equals(CLASS_2_NAME)) {
            DISPLAY_THRESHOLD = CLASS_2_THRESHOLD;
            boxPaint.setColor(CLASS_2_COLOR);
        } else if (className.equals(CLASS_3_NAME)) {
            DISPLAY_THRESHOLD = CLASS_3_THRESHOLD;
            boxPaint.setColor(CLASS_3_COLOR);
        } else if (className.equals(CLASS_4_NAME)) {
            DISPLAY_THRESHOLD = CLASS_4_THRESHOLD;
            boxPaint.setColor(CLASS_4_COLOR);
        } else if (className.equals(CLASS_5_NAME)) {
            DISPLAY_THRESHOLD = CLASS_5_THRESHOLD;
            boxPaint.setColor(CLASS_5_COLOR);
        } else if (className.equals(CLASS_6_NAME)) {
            DISPLAY_THRESHOLD = CLASS_6_THRESHOLD;
            boxPaint.setColor(CLASS_6_COLOR);
        } else if (className.equals(CLASS_7_NAME)) {
            DISPLAY_THRESHOLD = CLASS_7_THRESHOLD;
            boxPaint.setColor(CLASS_7_COLOR);
        }

      // Set the box stroke width
      boxPaint.setStrokeWidth(5.0f);


      // Determine whether or not we want to display this box based on the detected class
      if (shownDetections.contains(className) && confidence >= DISPLAY_THRESHOLD)
      {
          canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);
          borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.bottom, labelString);
      }
    }
      return conVals;
  }

  private boolean initialized = false;

  public synchronized void onFrame(
      final int w,
      final int h,
      final int rowStride,
      final int sensorOrienation,
      final byte[] frame,
      final long timestamp) {
    if (objectTracker == null && !initialized) {
      ObjectTracker.clearInstance();

      //logger.i("Initializing ObjectTracker: %dx%d", w, h);
      objectTracker = ObjectTracker.getInstance(w, h, rowStride, true);
      frameWidth = w;
      frameHeight = h;
      this.sensorOrientation = sensorOrienation;
      initialized = true;


      if (objectTracker == null) {
        String message =
            "Object tracking support not found. "
                + "See tensorflow/examples/android/README.md for details.";
        //Toast.makeText(context, message, Toast.LENGTH_LONG).show();
        //logger.e(message);
      }

    }

    if (objectTracker == null) {
      return;
    }

    objectTracker.nextFrame(frame, null, timestamp, null, true);

    // Clean up any objects not worth tracking any more.
    final LinkedList<TrackedRecognition> copyList =
        new LinkedList<TrackedRecognition>(trackedObjects);
    for (final TrackedRecognition recognition : copyList) {
      final ObjectTracker.TrackedObject trackedObject = recognition.trackedObject;
      final float correlation = trackedObject.getCurrentCorrelation();
      if (correlation < MIN_CORRELATION) {
        logger.v("Removing tracked object %s because NCC is %.2f", trackedObject, correlation);
        trackedObject.stopTracking();
        trackedObjects.remove(recognition);
        if (shownDetections.contains(recognition.title)) {
            numTracked--;
        }
        availableColors.add(recognition.color);
      }
    }
  }

  private void processResults(
      final long timestamp, final List<Recognition> results, final byte[] originalFrame) {
    final List<Triplet<String, Float, Recognition>> rectsToTrack = new LinkedList<Triplet<String, Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Triplet<String, Float, RectF>(result.getTitle(), result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE || (detectionFrameRect.width() > MAX_SIZE && detectionFrameRect.height() > MAX_SIZE)) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Triplet<String, Float, Recognition>(result.getTitle(), result.getConfidence(), result));
    }

    if (rectsToTrack.isEmpty()) {
      //logger.i("Nothing to track, aborting.");
        trackedObjects.clear();
        numTracked = 0;
        return;
    }

    if (objectTracker == null) {
      trackedObjects.clear();
      numTracked = 0;
      for (final Triplet<String,Float, Recognition> potential : rectsToTrack) {
        final TrackedRecognition trackedRecognition = new TrackedRecognition();
        trackedRecognition.detectionConfidence = potential.getSecond();
        trackedRecognition.location = new RectF(potential.getThird().getLocation());
        trackedRecognition.trackedObject = null;
        trackedRecognition.title = potential.getThird().getTitle();
        trackedRecognition.color = COLORS[trackedObjects.size()];
        trackedObjects.add(trackedRecognition);
        if (shownDetections.contains(trackedRecognition.title)) {
            numTracked++;
        }
        if (trackedObjects.size() >= COLORS.length) {
          break;
        }
      }
      return;
    }

    logger.i("%d rects to track", rectsToTrack.size());
    for (final Triplet<String,Float, Recognition> potential : rectsToTrack) {
      handleDetection(originalFrame, timestamp, potential);
    }
  }

  private void handleDetection(
      final byte[] frameCopy, final long timestamp, final Triplet<String, Float, Recognition> potential) {
    final ObjectTracker.TrackedObject potentialObject =
        objectTracker.trackObject(potential.getThird().getLocation(), timestamp, frameCopy);

    final float potentialCorrelation = potentialObject.getCurrentCorrelation();
    logger.v(
        "Tracked object went from %s to %s with correlation %.2f",
        potential.getThird(), potentialObject.getTrackedPositionInPreviewFrame(), potentialCorrelation);

    if (potentialCorrelation < MARGINAL_CORRELATION) {
      logger.v("Correlation too low to begin tracking %s.", potentialObject);
      potentialObject.stopTracking();
      return;
    }

    final List<TrackedRecognition> removeList = new LinkedList<TrackedRecognition>();

    float maxIntersect = 0.0f;

    // This is the current tracked object whose color we will take. If left null we'll take the
    // first one from the color queue.
    TrackedRecognition recogToReplace = null;

    // Look for intersections that will be overridden by this object or an intersection that would
    // prevent this one from being placed.
    for (final TrackedRecognition trackedRecognition : trackedObjects) {
      final RectF a = trackedRecognition.trackedObject.getTrackedPositionInPreviewFrame();
      final RectF b = potentialObject.getTrackedPositionInPreviewFrame();
      final RectF intersection = new RectF();
      final boolean intersects = intersection.setIntersect(a, b);

      final float intersectArea = intersection.width() * intersection.height();
      final float totalArea = a.width() * a.height() + b.width() * b.height() - intersectArea;
      final float intersectOverUnion = intersectArea / totalArea;

      // If there is an intersection with this currently tracked box above the maximum overlap
      // percentage allowed, either the new recognition needs to be dismissed or the old
      // recognition needs to be removed and possibly replaced with the new one.
      if (intersects && intersectOverUnion > MAX_OVERLAP) {
        if (potential.getSecond() < trackedRecognition.detectionConfidence
            && trackedRecognition.trackedObject.getCurrentCorrelation() > MARGINAL_CORRELATION) {
          // If track for the existing object is still going strong and the detection score was
          // good, reject this new object.
          potentialObject.stopTracking();
          return;
        } else {
          removeList.add(trackedRecognition);

          // Let the previously tracked object with max intersection amount donate its color to
          // the new object.
          if (intersectOverUnion > maxIntersect) {
            maxIntersect = intersectOverUnion;
            recogToReplace = trackedRecognition;
          }
        }
      }
    }

    // If we're already tracking the max object and no intersections were found to bump off,
    // pick the worst current tracked object to remove, if it's also worse than this candidate
    // object.
    if (availableColors.isEmpty() && removeList.isEmpty()) {
      for (final TrackedRecognition candidate : trackedObjects) {
        if (candidate.detectionConfidence < potential.getSecond()) {
          if (recogToReplace == null
              || candidate.detectionConfidence < recogToReplace.detectionConfidence) {
            // Save it so that we use this color for the new object.
            recogToReplace = candidate;
          }
        }
      }
      if (recogToReplace != null) {
        logger.v("Found non-intersecting object to remove.");
        removeList.add(recogToReplace);
      } else {
        logger.v("No non-intersecting object found to remove");
      }
    }

    // Remove everything that got intersected.
    for (final TrackedRecognition trackedRecognition : removeList) {
      logger.v(
          "Removing tracked object %s with detection confidence %.2f, correlation %.2f",
          trackedRecognition.trackedObject,
          trackedRecognition.detectionConfidence,
          trackedRecognition.trackedObject.getCurrentCorrelation());
      trackedRecognition.trackedObject.stopTracking();
      trackedObjects.remove(trackedRecognition);
      if (shownDetections.contains(trackedRecognition.title)) {
          numTracked--;
      }
      if (trackedRecognition != recogToReplace) {
        availableColors.add(trackedRecognition.color);
      }
    }

    if (recogToReplace == null && availableColors.isEmpty()) {
      logger.e("No room to track this object, aborting.");
      potentialObject.stopTracking();
      return;
    }

    // Finally safe to say we can track this object.
    logger.v(
        "Tracking object %s (%s) with detection confidence %.2f at position %s",
        potentialObject,
        potential.getThird().getTitle(),
        potential.getSecond(),
        potential.getThird().getLocation());
    final TrackedRecognition trackedRecognition = new TrackedRecognition();
    trackedRecognition.detectionConfidence = potential.getSecond();
    trackedRecognition.trackedObject = potentialObject;
    trackedRecognition.title = potential.getThird().getTitle();

    // Use the color from a replaced object before taking one from the color queue.
    trackedRecognition.color =
        recogToReplace != null ? recogToReplace.color : availableColors.poll();
    trackedObjects.add(trackedRecognition);
    if (shownDetections.contains(trackedRecognition.title)) {
        numTracked++;
    }
  }
}
