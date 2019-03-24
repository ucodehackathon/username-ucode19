import { Component, OnInit } from "@angular/core";
import { Router } from "@angular/router";

@Component({
  selector: "app-result",
  templateUrl: "./result.component.html",
  styleUrls: ["./result.component.scss"]
})
export class ResultComponent implements OnInit {
  constructor(private router: Router) {}

  ngOnInit() {}

  goToPredator() {
    window.open(
      "https://www.adidas.es/bota-de-futbol-predator-19.1-cesped-natural-seco/BC0551.html",
      "_blank"
    );
  }

  retry() {
    this.router.navigateByUrl("pose");
  }

  replay() {
    this.router.navigateByUrl("replay");
  }

  showAlert() {
    alert(
      "Gracias a las redes neuronales adversarias (GAN), establecemos un sistema en que varias redes compiten entre sí: una para generar una cara “aleatoria” y la otra para decidir si se ajusta al jugador al que más se parezca su tiro, consiguiendo al final una cara del usuario mezclada con la del jugador"
    );
  }
}
