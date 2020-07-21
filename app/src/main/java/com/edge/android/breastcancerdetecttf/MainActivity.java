package com.edge.android.breastcancerdetecttf;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import android.annotation.SuppressLint;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.List;
import java.lang.Math;


public class MainActivity extends AppCompatActivity {

    private Classifier mnistClassifier;
    private ClassifierNew newClassifier;
    ImageView imageView;
    Button chooseButton;
    Button detectButton;
    Button clearButton;
    TextView resultText;
    Boolean detectFlag = false;
    private static final int IMAGE_PICK_CODE = 1000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //Assigning variables to the UI elements and choosing the view
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.image_view);
        chooseButton = findViewById(R.id.choose_button);
        detectButton = findViewById(R.id.detect_button);
        clearButton = findViewById(R.id.clear_button);
        resultText = findViewById(R.id.result_text);
//        mnistClassifier = new Classifier(this);
        try {
            newClassifier = new ClassifierNew(this,1);
        } catch (IOException e) {
            e.printStackTrace();
        }

        //defining on click behavior
        chooseButton.setOnClickListener(new View.OnClickListener() {
            public void onClick (View v){
                selectImageFromGallery();
            }
        });

        clearButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                resultText.setVisibility(View.GONE);
                chooseButton.setVisibility(View.VISIBLE);
                imageView.setImageResource(R.drawable.newt);
                imageView.setVisibility(View.GONE);
                detectButton.setVisibility(View.GONE);
                v.setVisibility(View.GONE);
            }
        });

        detectButton.setOnClickListener(new View.OnClickListener(){
            @SuppressLint("ResourceType")
            @Override
            public void onClick(View v){
                if(detectFlag){
                    Bitmap bitmap = convertImageViewToBitmap(imageView);
//                    int width = bitmap.getWidth();
//                    int height = bitmap.getHeight();
//                    int imageSize = Math.min(width, height);
//                    Bitmap centreCroppedBitmap = BitmapUtil.centerCrop(bitmap, imageSize, imageSize);
//                    bitmap = Bitmap.createScaledBitmap(centreCroppedBitmap, 224, 224, true);


//                    int digit = mnistClassifier.classify(bitmap);
//                    String res = ""+(digit);
//                    if(digit==1) res = "Malignant";
//                    else res = "Benign";
//                    Toast toast1 = Toast.makeText(getApplicationContext(),res,Toast.LENGTH_SHORT);
//                    toast1.show();
                    String res ="";
                    List<ClassifierNew.Recognition> list = newClassifier.classify(bitmap,0);
                    for(ClassifierNew.Recognition entry: list){
                        res = res + entry.getTitle()+" "+ roundOff(entry.getConfidence())+"%\n\n";

                    }

                    resultText.setText(res);
                    resultText.setVisibility(View.VISIBLE);
//                    Toast toast1 = Toast.makeText(getApplicationContext(),res,Toast.LENGTH_SHORT);
//                    toast1.show();

                    v.setVisibility(View.GONE);
                }
                else{
                    Toast toast1 = Toast.makeText(getApplicationContext(), "No picture!", Toast.LENGTH_SHORT);
                    toast1.show();
                }
            }
        });


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        detectFlag = false;
        if(resultCode == RESULT_OK &&  requestCode == IMAGE_PICK_CODE){
            //set image to image view
            detectFlag = true;
            imageView = findViewById(R.id.image_view);
            imageView.setImageURI(data.getData());

            // Convert it to a bitmap which will be used for inference
            Bitmap imageBitmap = convertImageViewToBitmap(imageView);
            imageView.setImageBitmap(imageBitmap);
            //check proximity
            if(proximityScoreCheck(imageBitmap)<6){
                AlertDialog.Builder builder = new AlertDialog.Builder(this);
                builder.setMessage("This does not look like a Histopathological Image")
                        .setTitle("Are you Sure?")
                        .setPositiveButton("yes", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                // CONFIRM
                                // change UI visibility for the State.
                                imageView.setVisibility(View.VISIBLE);
                                detectButton.setVisibility(View.VISIBLE);
                                clearButton.setVisibility(View.VISIBLE);
                                resultText.setVisibility(View.GONE);
                                chooseButton.setVisibility(View.GONE);
                            }
                        })
                        .setNegativeButton("retry", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                // CANCEL
                                selectImageFromGallery();
                            }
                        });
                // Create the AlertDialog object and return it
                AlertDialog dialog = builder.create();
                dialog.show();
            }
            else{
                // change UI visibility for the State.
                imageView.setVisibility(View.VISIBLE);
                detectButton.setVisibility(View.VISIBLE);
                clearButton.setVisibility(View.VISIBLE);
                resultText.setVisibility(View.GONE);
                chooseButton.setVisibility(View.GONE);
            }
         }
    }

    private Bitmap convertImageViewToBitmap(ImageView iView){
        BitmapDrawable drawable = (BitmapDrawable) iView.getDrawable();
        Bitmap bitmap = drawable.getBitmap();
        return bitmap;
    }

    private int proximityScoreCheck(Bitmap bitmap){
        int score =0;
        double[] mean = {195.8494419642857, 163.7113887117348, 195.12751992984695};
        double[] stdDev = {12.595864429286115, 32.17582772629138, 13.283874465175264};
        double[] img = getPixelMean(bitmap);
        for(int color = 0; color<3;color+=1 ){
            double diff = Math.abs(img[color] - mean[color]);
            if(diff < stdDev[color]){
                score+=3;
            }
            else if (diff < 2*stdDev[color]){
                score+=2;
            }
            else if (diff < 3*stdDev[color]){
                score+=1;
            }
        }
        return score;
    }

    private double[] getPixelMean(Bitmap bitmap){
        bitmap = Bitmap.createScaledBitmap(bitmap,224,224,true);
        double redColors = 0;
        double greenColors = 0;
        double blueColors = 0;
        double pixelCount = 0;

        for (int y = 0; y < bitmap.getHeight(); y++)
        {
            for (int x = 0; x < bitmap.getWidth(); x++)
            {
                int c = bitmap.getPixel(x, y);
                pixelCount++;
                redColors += Color.red(c);
                greenColors += Color.green(c);
                blueColors += Color.blue(c);
            }
        }
        double red = (redColors/pixelCount);
        double green = (greenColors/pixelCount);
        double blue = (blueColors/pixelCount);
        double[] imgArr =  {red,green,blue};
        return imgArr;
    }

    private void selectImageFromGallery(){
        // Create a new intent to pick image for external app
        Intent imageIntent =new Intent(Intent.ACTION_PICK);
        // set type of file being excepted, image in our case, hence it opens gallery
        imageIntent.setType("image/*");
        // Run the activity for when we've picked the image
        if(imageIntent.resolveActivity(getPackageManager())!=null){
            startActivityForResult(imageIntent,IMAGE_PICK_CODE);
        }
    }

    private Float roundOff(Float val){
        val = val*10000;
        int temp = Math.round(val);
        val = (float) temp;
        val =val/100;
        return val;
    }

}
