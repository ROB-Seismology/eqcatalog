<?php
$inqtitre = "Did you feel the earthquake ? Report it here !";
$intro[1] = "You can help us to define the extent of shaking and
damage for earthquakes in Belgium, and you may provide specific
details about how your area may respond to future earthquakes.";

$intro[2] = "ROB scientists may use the information you enter in this form to
provide qualitative, quantitative, or graphical descriptions of
damage in their publications, but your personnal data will not be
used. If you would object to this possible usage of your data,
please do not fill out this form.";

$intro[3] = "<b>Your ZIP code is REQUIRED </b>to locate the intensity in
your area. All other identifiers (name, e-mail, phone, and location)
are optional, but may be critical for a more local evaluation.";

$intro[4] = "If you have information about more than one site, please
complete as many form as needed.";

$form1 = "QUESTIONNAIRE FOR THIS EVENT:";
$form2 = "Please make sure you are filling out this form for the right event.";
$form3 = "QUESTIONNAIRE FOR A NEW EVENT";
$form4 = "Date and time of the earthquake (approximate):";
$form5 = "Month:";
$amois = array("---", "Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.");
$form6 = "Day:";
$form7 = "Year:";
$form8[0] = "Time (HH:MM):";
$form8[1] = "AM";
$form8[2] = "PM";
$form9 = "Name:";
$form10 = "Phone:";
$form11 = "Your location when the earthquake occurred ?";
$form12 = "Since you are submitting this form for a
 non-listed earthquake, <b>please fill out the following
 information completely.</b> This will help us accurately
 locate this event.";
$form13 = "Street,<BR>Address:";
$form14 = "City:";
$form15 = "Zip Code:";
$form16 = "(REQUIRED!)";

$form17 = "Country:";
$v_country = array("BE","FR","NL","DE","LU","GB","??");
$a_country = array("Belgium","France","Nederland","Duitsland","GD du luxembourg","England","Other");
//$v_country = array("BE", "LU", "??");
//$a_country = array("Belgium", "Luxembourg", "Other");

/* A revoir */
$form19 = "Province:";
$v_province = array("Bruxelles", "Brabant Wallon", "Vlaams Brabant", "Hainaut", "Liège", "Namur", "Luxembourg", "Antwerpen", "West Vlaanderen", "Oost Vlaanderen", "Limburg", "other");
$a_province = array("Brussels", "Brabant Wallon", "Vlaams Brabant", "Hainaut", "Liège", "Namur", "Luxembourg", "Antwerpen", "West Vlaanderen", "Oost Vlaanderen", "Limburg", "Other");

$v_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "other");
$a_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Other");

$form20 = "While answering all these questions is optional, we
encourage you to fill out as many as possible so we can provide a
more accurate intensity estimate.";

$t_sit = "What was your situation during the earthquake ?";
$o_sit = array("No answer", "Inside", "Outside",
    "In stopped vehicle", "In moving vehicle",
    "Other");

$t_build = "If you were inside please select the type of building or
   structure:";
$o_build = array("No building", "Family Home", "Apartment Building",
    "Office Building/School",
    "Mobile Home with Permanent Foundation",
    "Trailer or Recr. Vehicle with No Foundation",
    "Other");

$form21 = "If you know the floor, please specify it :";
$form22 = "If other, please describe:";
$t_sleep = "Were you asleep during the earthquake ?";
$o_sleep = array("No", "Yes, and I slept through it", "Yes, but I woke up");

$form23 = "<b>Did you feel the earthquake ?</b>
(If you were asleep, did the earthquake wake you up?)";
$yes = "Yes";
$no = "No";
$t_ofelt = "Did others nearby feel the earthquake ?";
$o_ofelt = array("No answer / Don't know / Nobody else nearby",
    "No others felt it",
    "Some felt it, but most did not",
    "Most others felt it, but some did not",
    "Everyone or almost everyone felt it");
$form24 = "Your experience of the earthquake:";
$t_motion = "How would you best describe the ground shaking ?";
$o_motion = array("No description",
     "Not felt", "Weak", "Mild",
    "Moderate", "Strong", "Violent");

$form25 = "About how many seconds did the shaking last ?";

$t_reaction = "How would you best describe your reaction ?";
$o_reaction = array("No answer / Don't remember",
    "No reaction / Not felt",
    "Very little reaction", "Excitement",
    "Somewhat frightened", "Very frightened", "Extremely frightened");

$t_response = "How did you respond ? (Select one.)";
$o_response = array("No answer/Don't remember",
    "Took no action",
    "Moved to doorway",
    "Ducked and covered", "Ran outside", "Other");
$form26 = "If other, please describe:";

$t_stand = "Was it difficult to stand or walk ?";
$o_stand = array ("No answer / Did not try",
    "No", "Yes, difficult to stand",
    "Yes, I was fallen", "Yes, I was forcibly thrown to the ground");

$form27= "Earthquake effects on the furniture and the buildings :";

$t_sway = "Did you notice the swinging or swaying of doors or hanging objects ?";
$o_sway = array ("No answer / Did not look",
       "No","Yes, slight swinging","Yes, violent swinging");

$t_creak = "Did you notice creaking or other noises ?";
$o_creak = array ("No answer / Did not pay attention",
     "No","Yes, slight noise","Yes, loud noise");

$t_shelf = "Did objects rattle, topple over, or fall off shelves ?";
$o_shelf = array ("No answer / No shelves",
            "No","Rattled slightly","Rattled loudly",
			"A few toppled or fell off",
			"Many fell off",
			"Nearly everything fell off");

$t_picture = "Did pictures on walls move or get knocked askew ?";
$o_picture = array("No answer / No pictures",
            "No","Yes, but did not fall",
			"Yes, and some fell");
			
$t_furniture = "Did any furniture or appliances slide, tip over, or become displaced ?";
$o_furniture = array("No answer / No furniture",
               "No","Yes");


$t_heavy_appliance = "Was a heavy appliance (refrigerator or range) affected ?";
$o_heavy_appliance = array("No answer / No heavy appliance",
                  "No","Yes, some contents fell out",
                  "Yes, shifted by inches/cm ",
				  "Yes, shifted by a foot or more (>30cm)",
				  "Yes, overturned");

$t_walls = "Were free-standing walls or fences damaged ?";
$o_walls = array("No answer/No walls",
           "No","Yes, some were cracked",
           "Yes, some partially fell",
		   "Yes, some fell completely");

				  
 
$t_d_text ="If you were inside, was there any damage to the building ? 
           Check all that apply."; 
$o_d_text =array("No damage",
                 "Hairline cracks in walls",
				 "A few large cracks in walls",
				 "Many large cracks in walls",
				 "Ceiling tiles or lighting fixtures fell",
				 "Cracks in chimney",
				 "One or several cracked windows",
				 "Many windows cracked or some broken out",
				 "Masonry fell from block or brick wall(s)",
				 "Old chimney, major damage or fell down",
				 "Modern chimney, major damage or fell down",
				 "Outside wall(s) tilted over or collapsed completely",
				 "Separation of porch, balcony, or other addition from building",
	             "Building shifted over foundation"); 
 
 
$form28="If you know the type (wood, brick, etc.) and/or
   the height (in floors) of building please indicate here:";
 
 
$form29="Additional Comments:";
$form30="If you have some complementary observations, use the next box. <BR> (Do not ask questions here. Use the email)";

$form31="To submit your completed form, press this button :"; 
$form32="Click here to clear the entire form and start over :"; 

$form33="Thank you for completing this form. Please report any problems or corrections
   to ";
$form34=" Modifications : Royal Observatory of Belgium - 2000/".date("Y")." ";
$form35="Verification Code";

$warn0 = "WARNING ";
$warn1 = "is not an accepted country in this inquiry.";
$warn2 = "Please consult the website for this country.";
$warn3 = "is not a valid e-mail address.";
$warn4 = $form15;
$warn5 = "Not found in our database.";
$warn6 = "Searching for the village : ";
$warn7 = $form14;
$warn8 = "You are not allowed to send two forms for the same event and for the same locality.";
$warn9 = "Zip code not found in the database.";
$warn10 = "Database connection failed.";
$warn11 = "City not found in the database.";
$warn12 = "The verification code is not valid.";
$warn13 = "Please enter the approximate hour of the event (HH:MM).";
$warn14 = "Please enter your zip code.";
$warn15 = "Please enter the verification code."; 
$warn16 = "Please select your zip code.";

$rep0 = "Macroseismic Inquiry :";
$rep1 = "Thank you for completing this form.";
$rep2 = "We have registered your answer.";
$rep3 = "The locality where you have experienced the earthquake :";
$rep4 = "Your e-mail address :";
$rep5 = "Intensity calculated for your village based on your answers :";
$rep6 = "Close";

$secu1 = "Type the characters you see in the picture below";
$secu2 = "Letters are not case-sensitive ";

$t_noise = "Did you hear a noise ?";
$o_noise = array("No",
    "Yes, Light and brief noise",
    "Yes, Light and prolonged noise",
    "Yes, Strong and brief noise",
    "Yes, Strong and prolonged noise");
?>
