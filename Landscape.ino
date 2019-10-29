 //Declare pin functions on RedBoard
#define stp 5
#define dir 6
#define MS1 10
#define MS2 9
#define EN  8

//Declare variables for functions
char user_input;
int x;
int y;
int state;
int count = 0;


void setup() {
  // put your setup code here, to run once:
  pinMode(stp, OUTPUT);
  pinMode(dir, OUTPUT);
  pinMode(MS1, OUTPUT);
  pinMode(MS2, OUTPUT);
  pinMode(EN, OUTPUT);
  pinMode(LED_BUILTIN,OUTPUT);
  resetEDPins(); //Set step, direction, microstep and enable pins to default states
  Serial.begin(9600); //Open Serial connection for debugging
  Serial.println("Begin motor control");
  Serial.println();
}

void loop() {
  // put your main code here, to run repeatedly:
  while (!Serial.available()) {}
  while(Serial.available()){
      String user_input = Serial.readString(); //Read user input and trigger appropriate function
      digitalWrite(EN, LOW); //Pull enable pin low to allow motor control
      if (user_input == "1")
      {
         SmallStepMode();
      }
      else if(user_input == "2")
      {
        ReverseStepDefault();
      }
      else if (user_input == "3"){
        count = 0;
      }
      else
      {
        Serial.println("Invalid option entered.");
      } 
  }
   if (count == 10){
     digitalWrite(EN, LOW); 
        ReverseStepDefault();
        count = 0;
        resetEDPins();
      }
    Serial.println(count);
}

//Reverse default microstep mode function
void ReverseStepDefault()
{
//  Serial.println("Moving in reverse at default step mode.");
  digitalWrite(dir, HIGH); //Pull direction pin high to move in "reverse"
  for(x= 0; x<400; x++)  //Loop the stepping enough times for motion to be visible
  {
    digitalWrite(stp,HIGH); //Trigger one step
    delay(1);
    digitalWrite(stp,LOW); //Pull step pin low so it can be triggered again
    delay(1);
  }
  delay(5000);
}

// 1/8th microstep foward mode function
void SmallStepMode()
{
  Serial.println("Stepping at 1/8th microstep mode.");
  digitalWrite(dir, LOW); //Pull direction pin low to move "forward"
  digitalWrite(MS1, HIGH); //Pull MS1, and MS2 high to set logic to 1/8th microstep resolution
  digitalWrite(MS2, HIGH);
  for(x= 0; x<100; x++)  //Loop the forward stepping enough times for motion to be visible
  {
    digitalWrite(stp,HIGH); //Trigger one step forward
    delay(1);
    digitalWrite(stp,LOW); //Pull step pin low so it can be triggered again
    delay(1);
  }
  count = count+1;
}

//Reset Easy Driver pins to default states
void resetEDPins()
{
  digitalWrite(stp, LOW);
  digitalWrite(dir, LOW);
  digitalWrite(MS1, LOW);
  digitalWrite(MS2, LOW);
  digitalWrite(EN, HIGH);
}
