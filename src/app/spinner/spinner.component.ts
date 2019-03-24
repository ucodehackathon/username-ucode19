import { AfterViewInit, Component } from "@angular/core";
import { Router } from "@angular/router";

@Component({
  selector: "app-spinner",
  templateUrl: "./spinner.component.html",
  styleUrls: ["./spinner.component.scss"]
})
export class SpinnerComponent implements AfterViewInit {
  constructor(private router: Router) {}

  ngAfterViewInit() {
    setTimeout(() => {
      this.router.navigateByUrl("result");
    }, 5000);
  }
}
