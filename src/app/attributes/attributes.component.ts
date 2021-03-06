import { Component, OnInit } from "@angular/core";
import { Router } from "@angular/router";

@Component({
  selector: "app-attributes",
  templateUrl: "./attributes.component.html",
  styleUrls: ["./attributes.component.scss"]
})
export class AttributesComponent implements OnInit {
  constructor(private router: Router) {}

  ngOnInit() {}

  goToSteps() {
    this.router.navigateByUrl("steps");
  }
}
