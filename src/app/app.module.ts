import { NgModule } from "@angular/core";
import { MatButtonModule } from "@angular/material/button";
import { MatCardModule } from "@angular/material/card";
import { MatGridListModule } from "@angular/material/grid-list";
import { MatListModule } from "@angular/material/list";
import { MatToolbarModule } from "@angular/material/toolbar";
import { BrowserModule } from "@angular/platform-browser";
import { RouterModule } from "@angular/router";
import { StarRatingModule } from "angular-star-rating";
import { WebcamModule } from "ngx-webcam";
import { AppComponent } from "./app.component";
import { AttributesComponent } from "./attributes/attributes.component";
import { CameraComponent } from "./camera/camera.component";
import { HomeComponent } from "./home/home.component";
import { PoseComponent } from "./pose/pose.component";
import { ResultComponent } from "./result/result.component";
import { ScannerComponent } from "./scanner/scanner.component";
import { SpinnerComponent } from "./spinner/spinner.component";
import { StepsComponent } from "./steps/steps.component";
import { VideoCaptureComponent } from "./video-capture/video-capture.component";

@NgModule({
  declarations: [
    AppComponent,
    CameraComponent,
    HomeComponent,
    ScannerComponent,
    AttributesComponent,
    StepsComponent,
    PoseComponent,
    VideoCaptureComponent,
    SpinnerComponent,
    ResultComponent
  ],
  imports: [
    BrowserModule,
    RouterModule.forRoot(
      [
        { path: "home", component: HomeComponent },
        { path: "scanner", component: ScannerComponent },
        { path: "attributes", component: AttributesComponent },
        { path: "steps", component: StepsComponent },
        { path: "pose", component: PoseComponent },
        { path: "video-capture", component: VideoCaptureComponent },
        { path: "spinner", component: SpinnerComponent },
        { path: "result", component: ResultComponent },
        { path: "camera", component: CameraComponent },
        { path: "**", redirectTo: "home", pathMatch: "full" }
      ],
      { onSameUrlNavigation: "reload" }
    ),
    MatButtonModule,
    MatGridListModule,
    MatCardModule,
    MatToolbarModule,
    MatListModule,
    StarRatingModule.forRoot(),
    WebcamModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
