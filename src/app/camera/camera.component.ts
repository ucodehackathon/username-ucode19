import { AfterViewInit, Component } from "@angular/core";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as posenet from "@tensorflow-models/posenet";
import dat from "dat.gui";
import * as html2canvas from "html2canvas";
import Stats from "stats.js";
import { drawBoundingBox, drawKeypoints, drawSkeleton } from "./demo_util";
import * as MomentumSlider from "./momentum-slider";

export interface Tile {
  color: string;
  cols: number;
  rows: number;
  text: string;
}

@Component({
  selector: "app-camera",
  templateUrl: "./camera.component.html",
  styleUrls: ["./camera.component.scss"]
})
export class CameraComponent implements AfterViewInit {
  countdownActive = false;

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

  videoWidth = 540;
  videoHeight = 960;
  stats = new Stats();

  running = false;
  timer = null;
  seconds = 0;
  secondsInitial = 40;
  root;
  container;
  button;
  countdownValue: any = 3;

  // Initializing the slider
  ms;

  showVideo = true;
  showPredictions = false;

  tiles: Tile[] = [
    { text: "One", cols: 4, rows: 4, color: "lightblue" },
    { text: "Two", cols: 1, rows: 2, color: "lightgreen" },
    { text: "Three", cols: 1, rows: 1, color: "lightpink" },
    { text: "Four", cols: 2, rows: 1, color: "#DDBDF1" }
  ];

  modelPromise = cocoSsd.load("lite_mobilenet_v2");
  screenshot;

  predictionInProgress = false;

  startCountdown() {
    this.countdownActive = true;
    this.countdownValue = 3;
    this.countdownTimeout();
  }

  countdownTimeout() {
    setTimeout(() => {
      this.countdownValue -= 1;
      if (this.countdownValue > 0) {
        this.countdownTimeout();
      } else {
      }
    }, 1000);
  }

  ngAfterViewInit() {
    // Variables to use later
    this.root = document.documentElement;
    this.container = document.querySelector(".container");
    this.button = document.querySelector(".button");

    // Initializing the slider
    this.ms = new MomentumSlider({
      el: this.container, // HTML element to append the slider
      range: [5, 40], // Generar los elementos del slider usando el rango de números definido
      loop: 2, // Hacer infinito el slider, añadiendo 2 elementos extra en cada extremo
      style: {
        // Estilos para interpolar
        // El primer valor corresponde a los elementos adyacentes
        // El segundo valor corresponde al elemento actual
        transform: [{ scale: [0.4, 1] }],
        opacity: [0.3, 1]
      }
    });

    // Simple toggle functionality
    this.button.addEventListener("click", () => {
      if (this.running) {
        this.stop();
      } else {
        this.start();
      }
      this.running = !this.running;
    });
  }

  start() {
    this.showVideo = true;
    // Disable the slider during countdown
    this.ms.disable();
    // Get current slide index, and set initial values
    this.seconds = this.ms.getCurrentIndex() + this.countdownValue;
    this.countdownValue = this.secondsInitial = this.seconds;
    this.root.style.setProperty("--progress", "0");
    // Add class to trigger CSS transitions for `running` state
    this.container.classList.add("container--running");
    // Set interval to update the component every second
    const that = this;
    this.timer = setInterval(function() {
      // Update values
      that.countdownValue = --that.seconds;
      that.root.style.setProperty(
        "--progress",
        (
          ((that.secondsInitial - that.seconds) / that.secondsInitial) *
          100
        ).toString()
      );

      if (that.countdownValue === 2) {
        setTimeout(() => {
          navigator.getUserMedia = navigator.getUserMedia;
          that.bindPage();
          // setTimeout(() => {
          //   that.showVideo = false;
          // }, 6000);
          // PANTALLASOOOOOOOO
          setTimeout(() => {
            that.showVideo = false;
          }, 4350);
        }, 1000);
      }
      // Stop countdown if it's finished
      if (!that.seconds) {
        that.stop();
        setTimeout(() => {
          that.running = false;
        }, 500);
      }
    }, 900);
  }

  stop() {
    // Enable slider
    this.ms.enable();
    // Clear interval
    clearInterval(this.timer);
    // Reset progress
    this.root.style.setProperty("--progress", "100");
    // Remove `running` state
    this.container.classList.remove("container--running");
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
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: mobile ? undefined : this.videoWidth,
        height: mobile ? undefined : this.videoHeight
      }
    });
    video.srcObject = stream;
    // video.src = "assets/videos/vid1-00-04.mp4";
    // video.src = 'https://www.youtube.com/watch?v=uZsnr4No36I';

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

    // The single-pose algorithm is faster and simpler but requires only one
    // person to be in the frame or results will be innaccurate. Multi-pose works
    // for more than 1 person
    const algorithmController = gui.add(this.guiState, "algorithm", [
      "single-pose",
      "multi-pose"
    ]);

    // The input parameters have the most effect on accuracy and speed of the
    // network
    const input = gui.addFolder("Input");
    // Architecture: there are a few PoseNet models varying in size and
    // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
    // fastest, but least accurate.
    const architectureController = input.add(
      this.guiState.input,
      "mobileNetArchitecture",
      ["1.01", "1.00", "0.75", "0.50"]
    );
    // Output stride:  Internally, this parameter affects the height and width of
    // the layers in the neural network. The lower the value of the output stride
    // the higher the accuracy but slower the speed, the higher the value the
    // faster the speed but lower the accuracy.
    input.add(this.guiState.input, "outputStride", [8, 16, 32]);
    // Image scale factor: What to scale the image by before feeding it through
    // the network.
    input
      .add(this.guiState.input, "imageScaleFactor")
      .min(0.2)
      .max(1.0);
    //input.open();

    // Pose confidence: the overall confidence in the estimation of a person's
    // pose (i.e. a person detected in a frame)
    // Min part confidence: the confidence that a particular estimated keypoint
    // position is accurate (i.e. the elbow's position)
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

  /**
   * Sets up a frames per second panel on the top-left of the window
   */
  setupFPS() {
    this.stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(this.stats.dom);
  }

  /**
   * Feeds an image to posenet to estimate poses - this is where the magic
   * happens. This function loops with a requestAnimationFrame method.
   */
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
    this.stats.begin();

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

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
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

    console.log("this.showVideo: ", this.showVideo);
    if (this.showVideo) {
      try {
        // if (!this.predictionInProgress) {
        //   html2canvas(document.querySelector("#output")).then(canvas => {
        //     this.screenshot = canvas.toDataURL();
        //     setTimeout(() => {
        //       this.runPredictions();
        //       this.showPredictions = true;
        //     }, 500);
        //   });
        // }
        requestAnimationFrame(this.poseDetectionFrame(
          video,
          flipHorizontal,
          ctx,
          guiState
        ) as any);
      } catch (e) {}
    } else {
      html2canvas(document.querySelector("#output")).then(canvas => {
        this.screenshot = canvas.toDataURL();
        setTimeout(() => {
          this.runPredictions();
          this.showPredictions = true;
        }, 500);
      });
    }
  }

  /**
   * Kicks off the demo by loading the posenet model, finding and loading
   * available camera devices, and setting off the detectPoseInRealTime function.
   */
  async bindPage() {
    // Load the PoseNet model weights with architecture 0.75
    const net = await posenet.load(0.75);

    // document.getElementById("loading").style.display = "none";
    // document.getElementById("main").style.display = "block";

    let video;

    try {
      video = await this.loadVideo();
    } catch (e) {
      // const info = document.getElementById("info");
      // info.textContent =
      //   "this browser does not support video capture," +
      //   "or this device does not have a camera";
      // info.style.display = "block";
      throw e;
    }

    this.setupGui([], net);
    this.setupFPS();
    this.detectPoseInRealTime(video, net);
  }

  async runPredictions() {
    this.predictionInProgress = true;
    const image: any = document.querySelector("#image");
    const model = await this.modelPromise;
    console.log("model loaded");
    console.time("predict1");
    // console.log("image: ", image);
    const result = await model.detect(image);
    // console.log("result: ", result);
    console.timeEnd("predict1");

    const c: any = document.querySelector("#canvas");
    // console.log("c: ", c);
    const context = c.getContext("2d");
    context.drawImage(image, 0, 0);
    context.font = "20px Arial";

    c.width = this.videoWidth;
    c.height = this.videoHeight;

    console.log("number of detections: ", result.length);
    for (let i = 0; i < result.length; i++) {
      context.beginPath();
      context.rect(...result[i].bbox);
      context.lineWidth = 1;
      context.strokeStyle = "black";
      context.fillStyle = "black";
      context.stroke();
      context.fillText(
        result[i].score.toFixed(3) + " " + result[i].class,
        result[i].bbox[0],
        result[i].bbox[1] > 10 ? result[i].bbox[1] - 5 : 10
      );
    }

    this.predictionInProgress = false;
  }

  // async runPredictions() {
  //   const image: any = document.querySelector("#image");
  //   const model = await this.modelPromise;
  //   console.log("model loaded");
  //   console.time("predict1");
  //   // console.log("image: ", image);
  //   const result = await model.detect(image);
  //   console.log('result: ', result);
  //   console.timeEnd("predict1");

  //   const c: any = document.querySelector("#canvas");
  //   // console.log("c: ", c);
  //   const context = c.getContext("2d");
  //   context.drawImage(image, 0, 0);
  //   context.font = "20px Arial";

  //   c.width = this.videoWidth;
  //   c.height = this.videoHeight;

  //   console.log("number of detections: ", result.length);
  //   for (let i = 0; i < result.length; i++) {
  //     context.beginPath();
  //     context.rect(...result[i].bbox);
  //     context.lineWidth = 1;
  //     context.strokeStyle = "black";
  //     context.fillStyle = "black";
  //     context.stroke();
  //     context.fillText(
  //       result[i].score.toFixed(3) + " " + result[i].class,
  //       result[i].bbox[0],
  //       result[i].bbox[1] > 10 ? result[i].bbox[1] - 5 : 10
  //     );
  //   }
  // }
}
