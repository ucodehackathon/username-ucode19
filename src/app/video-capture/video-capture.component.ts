import { AfterViewInit, Component } from "@angular/core";
import { Router } from "@angular/router";
import * as posenet from "@tensorflow-models/posenet";
import * as tf from "@tensorflow/tfjs";
import { loadLayersModel } from "@tensorflow/tfjs";
import dat from "dat.gui";
import * as html2canvas from "html2canvas";
import Stats from "stats.js";
import { drawBoundingBox, drawKeypoints, drawSkeleton } from "./demo_util";

declare var MediaRecorder: any;

@Component({
  selector: "app-video-capture",
  templateUrl: "./video-capture.component.html",
  styleUrls: ["./video-capture.component.scss"]
})
export class VideoCaptureComponent implements AfterViewInit {
  guiState: any = {
    algorithm: "multi-pose",
    input: {
      mobileNetArchitecture: this.isMobile() ? "0.50" : "0.75",
      outputStride: 16,
      imageScaleFactor: 0.5
    },
    singlePoseDetection: {
      minPoseConfidence: 0.1,
      minPartConfidence: 0.5
    },
    multiPoseDetection: {
      maxPoseDetections: 5,
      minPoseConfidence: 0.15,
      minPartConfidence: 0.1,
      nmsRadius: 30.0
    },
    output: {
      showVideo: true,
      showSkeleton: true,
      showPoints: true,
      showBoundingBox: false
    },
    net: null
  };

  videoWidth = 359;
  videoHeight = 718;
  stats = new Stats();

  showingVideo = true;
  videoDuration = 6000;
  screenshot;

  stream: MediaStream;

  shouldStop = false;
  mediaRecorder;

  constructor(private router: Router) {}

  ngAfterViewInit() {
    this.capture();
  }

  isAndroid() {
    return /Android/i.test(navigator.userAgent);
  }

  isiOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
  }

  isMobile() {
    return this.isAndroid() || this.isiOS();
  }

  async setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Browser API navigator.mediaDevices.getUserMedia not available"
      );
    }

    const video: any = document.getElementById("video");
    video.width = this.videoWidth;
    video.height = this.videoHeight;

    const mobile = this.isMobile();
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: mobile ? undefined : this.videoWidth,
        height: mobile ? undefined : this.videoHeight
      }
    });
    video.srcObject = this.stream;
    // video.src = "assets/videos/vid1-00-04.mp4";

    const options = { mimeType: "video/webm" };
    const recordedChunks = [];
    let stopped = false;

    this.mediaRecorder = new MediaRecorder(this.stream, options);

    this.mediaRecorder.addEventListener("dataavailable", e => {
      console.log("e: ", e);
      if (e.data.size > 0) {
        recordedChunks.push(e.data);
      }

      if (this.shouldStop === true && stopped === false) {
        stopped = true;
      }
    });

    this.mediaRecorder.addEventListener("stop", async () => {
      // downloadLink.href = URL.createObjectURL(new Blob(recordedChunks));
      // downloadLink.download = "acetest.webm";

      console.log("FINNNNNN", new Blob(recordedChunks));

      const blob = new Blob(recordedChunks);
      // do something with this blob
      const vidURL = URL.createObjectURL(blob);
      const vid = document.createElement("video");
      vid.src = vidURL;
      vid.height = 1920;
      vid.width = 1080;

      console.log("recordedChunks: ", recordedChunks);

      const MODEL_URL = "assets/model/model.json";

      const model = await loadLayersModel(MODEL_URL);

      const tensorArray = tf.tensor(recordedChunks);

      const tfFromPixels = tf.browser.fromPixels(vid);
      tfFromPixels.shape.splice(0,3);
      tfFromPixels.shape.push(null);
      tfFromPixels.shape.push(240);
      tfFromPixels.shape.push(240);
      tfFromPixels.shape.push(23);
      console.log('tfFromPixels: ', tfFromPixels);

      const asdf = model.predict(tfFromPixels, { verbose: true });
      console.log("resultExect: ", asdf);
    });

    console.log("mediaRecorder: ", this.mediaRecorder);

    this.mediaRecorder.start(33.33);

    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
    });
  }

  async loadVideo() {
    const video: any = await this.setupCamera();
    video.play();

    return video;
  }

  /**
   * Sets up dat.gui controller on the top-right of the window
   */
  setupGui(cameras, net) {
    this.guiState.net = net;

    if (cameras.length > 0) {
      this.guiState.camera = cameras[0].deviceId;
    }

    const gui = new dat.GUI({ width: 300 });
    const algorithmController = gui.add(this.guiState, "algorithm", [
      "single-pose",
      "multi-pose"
    ]);

    const input = gui.addFolder("Input");

    const architectureController = input.add(
      this.guiState.input,
      "mobileNetArchitecture",
      ["1.01", "1.00", "0.75", "0.50"]
    );
    input.add(this.guiState.input, "outputStride", [8, 16, 32]);
    input
      .add(this.guiState.input, "imageScaleFactor")
      .min(0.2)
      .max(1.0);
    const single = gui.addFolder("Single Pose Detection");
    single.add(
      this.guiState.singlePoseDetection,
      "minPoseConfidence",
      0.0,
      1.0
    );
    single.add(
      this.guiState.singlePoseDetection,
      "minPartConfidence",
      0.0,
      1.0
    );

    const multi = gui.addFolder("Multi Pose Detection");
    multi
      .add(this.guiState.multiPoseDetection, "maxPoseDetections")
      .min(1)
      .max(20)
      .step(1);
    multi.add(this.guiState.multiPoseDetection, "minPoseConfidence", 0.0, 1.0);
    multi.add(this.guiState.multiPoseDetection, "minPartConfidence", 0.0, 1.0);
    // nms Radius: controls the minimum distance between poses that are returned
    // defaults to 20, which is probably fine for most use cases
    multi
      .add(this.guiState.multiPoseDetection, "nmsRadius")
      .min(0.0)
      .max(40.0);
    //multi.open();

    const output = gui.addFolder("Output");
    output.add(this.guiState.output, "showVideo");
    output.add(this.guiState.output, "showSkeleton");
    output.add(this.guiState.output, "showPoints");
    output.add(this.guiState.output, "showBoundingBox");
    //output.open();

    architectureController.onChange(architecture => {
      this.guiState.changeToArchitecture = architecture;
    });

    algorithmController.onChange(value => {
      switch (this.guiState.algorithm) {
        case "single-pose":
          multi.close();
          single.open();
          break;
        case "multi-pose":
          single.close();
          multi.open();
          break;
      }
    });

    gui.close();
  }

  setupFPS() {
    //this.stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    //document.body.appendChild(this.stats.dom);
  }

  detectPoseInRealTime(video, net) {
    const canvas: any = document.getElementById("output");
    const ctx = canvas.getContext("2d");
    // since images are being fed from a webcam
    const flipHorizontal = true;

    canvas.width = this.videoWidth;
    canvas.height = this.videoHeight;

    this.poseDetectionFrame(video, flipHorizontal, ctx, this.guiState);
  }

  async poseDetectionFrame(video, flipHorizontal, ctx, guiState) {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
      // version
      guiState.net = await posenet.load(guiState.changeToArchitecture);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    // this.stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
      case "single-pose":
        const pose = await guiState.net.estimateSinglePose(
          video,
          imageScaleFactor,
          flipHorizontal,
          outputStride
        );
        poses.push(pose);

        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case "multi-pose":
        poses = await guiState.net.estimateMultiplePoses(
          video,
          imageScaleFactor,
          flipHorizontal,
          outputStride,
          guiState.multiPoseDetection.maxPoseDetections,
          guiState.multiPoseDetection.minPartConfidence,
          guiState.multiPoseDetection.nmsRadius
        );

        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, this.videoWidth, this.videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-this.videoWidth, 0);
      ctx.drawImage(video, 0, 0, this.videoWidth, this.videoHeight);
      ctx.restore();
    }

    poses.forEach(({ score, keypoints }) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      }
    });

    // End monitoring code for frames per second
    this.stats.end();

    try {
      if (this.showingVideo) {
        requestAnimationFrame(this.poseDetectionFrame(
          video,
          flipHorizontal,
          ctx,
          guiState
        ) as any);
      } else {
        html2canvas(document.querySelector("#output")).then(canvas => {
          this.screenshot = canvas.toDataURL();
          this.shouldStop = true;

          this.mediaRecorder.stop();
          setTimeout(() => {
            // this.router.navigateByUrl("spinner");
          }, 1500);
        });
      }
    } catch (e) {}
  }

  async capture() {
    this.showingVideo = true;
    setTimeout(() => {
      this.showingVideo = false;
    }, this.videoDuration);
    // Load the PoseNet model weights with architecture 0.75
    const net = await posenet.load(0.75);

    let video;

    try {
      video = await this.loadVideo();
    } catch (e) {
      throw e;
    }

    this.setupGui([], net);
    this.setupFPS();
    this.detectPoseInRealTime(video, net);
  }
}
