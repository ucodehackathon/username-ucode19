import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-steps',
  templateUrl: './steps.component.html',
  styleUrls: ['./steps.component.scss']
})
export class StepsComponent implements OnInit {

  constructor(private router: Router) { }

  ngOnInit() {
  }

  goToPose(){
    this.router.navigateByUrl('pose');
  }

}
