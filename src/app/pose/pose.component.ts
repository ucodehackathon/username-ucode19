import { AfterViewInit, Component } from "@angular/core";
import { Router } from "@angular/router";
import * as MomentumSlider from "./momentum-slider";

@Component({
  selector: "app-pose",
  templateUrl: "./pose.component.html",
  styleUrls: ["./pose.component.scss"]
})
export class PoseComponent implements AfterViewInit {
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

  constructor(private router: Router) {}

  ngAfterViewInit() {
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

    setTimeout(() => {
      this.click();
    }, 500);
  }

  click() {
    if (this.running) {
      this.stop();
    } else {
      this.start();
    }
    this.running = !this.running;
  }

  start() {
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
    this.timer = setInterval(() => {
      // Update values
      that.countdownValue = --that.seconds;
      that.root.style.setProperty(
        "--progress",
        (
          ((that.secondsInitial - that.seconds) / that.secondsInitial) *
          100
        ).toString()
      );

      // Stop countdown if it's finished
      if (!that.seconds) {
        that.stop();
        setTimeout(() => {
          that.running = false;
          this.router.navigateByUrl("video-capture");
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
}
