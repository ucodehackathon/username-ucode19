import { Component, OnInit } from "@angular/core";

@Component({
  selector: "app-result",
  templateUrl: "./result.component.html",
  styleUrls: ["./result.component.scss"]
})
export class ResultComponent implements OnInit {
  constructor() {}

  ngOnInit() {}

  goToPredator() {
    window.open(
      "https://www.adidas.es/bota-de-futbol-predator-19.1-cesped-natural-seco/BC0551.html",
      "_blank"
    );
  }
  showAlert(msg) {
    alert(msg);
  }
}
