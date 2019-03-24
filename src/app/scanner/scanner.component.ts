import { Component, OnInit } from "@angular/core";
import { Router } from "@angular/router";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { WebcamImage } from "ngx-webcam";
import { Observable, Subject } from "rxjs";

@Component({
  selector: "app-scanner",
  templateUrl: "./scanner.component.html",
  styleUrls: ["./scanner.component.scss"]
})
export class ScannerComponent implements OnInit {
  imageTaken = false;
  webcamImage: WebcamImage;

  private trigger: Subject<void> = new Subject<void>();

  modelPromise = cocoSsd.load("lite_mobilenet_v2");

  constructor(private router: Router) {}

  ngOnInit() {}

  takeImage() {
    this.imageTaken = true;
    this.triggerSnapshot();
  }

  handleImage(webcamImage: WebcamImage): void {
    console.info("received webcam image", webcamImage);
    this.webcamImage = webcamImage;
  }

  triggerSnapshot(): void {
    this.trigger.next();
    setTimeout(() => {
      this.runPredictions();
    }, 500);
  }

  get triggerObservable(): Observable<void> {
    return this.trigger.asObservable();
  }

  goToAttributes() {
    this.router.navigateByUrl("attributes");
  }

  async runPredictions() {
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

    c.width = 327;
    c.height = 755;

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
  }

  retake() {
    this.router.navigateByUrl("scanner");
  }
}
