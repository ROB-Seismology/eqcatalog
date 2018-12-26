<?php
$inqtitre = "Haben Sie ein Erdbeben gesp�rt ?  Bitte geben Sie Ihren Bericht hier ein :";
$intro[1] = "Sie k�nnen uns helfen die Ausdehnung der Ersch�tterungen und 
Sch�den von Erdbeben zu erfassen. Sie helfen damit, 
Einzelheiten der Wirkung zuk�nftiger Beben in Ihrer Region abzusch�tzen und die Erdbebenvorsorge zu verbessern.";

$intro[2] = "KSB und BNS Wissenschaftler k�nnen die Informationen, die Sie in diesen
Fragebogen eingeben, benutzen, um in Ihren Publikationen die Wirkung
von Erdbeben qualitativ, quantitativ oder graphisch zu beschreiben.
Ihre pers�nlichen Daten werden nicht weiterverwendet. Wenn Sie dieser
Verwendung Ihrer Angaben nicht zustimmen, dann f�llen Sie den Fragebogen
bitte nicht aus.";

$intro[3] = "<b>Ihre POSTLEITZAHL ist ERFORDERLICH </b>, um die Intensit�t in
Ihrer Gegend festlegen zu k�nnen. Alle anderen Angaben (Name, e-mail, 
Telefonnummer und Ortsangabe) sind optional, k�nnen aber f�r 
Detailuntersuchungen wichtig sein.";

$intro[4] = "Wenn Sie Informationen von mehr als einem Ort haben, so f�llen Sie bitte f�r jeden Ort ein eigenes Formular aus.";

$form1 = "FRAGEBOGEN F�R:";
$form2 = "Vergewissern sie sich bitte das dem Ereignis entsprechende Formular auszuf�llen.";
$form3 = "FRAGEBOGEN F�R EIN NEUES ERDBEBEN";
$form4 = "Datum und Zeit des Erdbebens (ungef�hr):";
$form5 = "Monat:";
$amois = array("---", "Jan.", "Feb.", "M�r.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.", "Sep.", "Okt.", "Nov.", "Dez.");
$form6 = "Tag:";
$form7 = "Jahr:";
$form8[0] = "Uhrzeit (HH:MM):";
$form8[1] = "AM";
$form8[2] = "PM";
$form9 = "Name:";
$form10 = "Telefon:";
$form11 = "Ihr Aufenthaltsort zu dem Zeitpunkt, als Sie das Beben wahrgenommen haben ?";
$form12 = "Da Sie das Formular f�r ein noch nicht aufgef�hrtes Erdbeben ausf�llen, 
bitten wir Sie die folgenden Informationen komplett anzugeben. Dies wird 
uns helfen das Erdbeben genauestens zu lokalisieren.";
$form13 = "Stra�e,<BR>Anschrift:";
$form14 = "Stadt:";
$form15 = "PLZ:";
$form16 = "(ERFORDERLICH!)";

$form17 = "Land:";
$v_country = array("BE","FR","NL","DE","LU","GB","??");
$a_country = array("Belgien","Frankreich","Niederlande","Deutschland","Luxemburg","Gro�britannien","Andere");
//$v_country = array("BE", "LU", "??");
//$a_country = array("Belg�e", "Luxemburg", "Andere");

/* a v�rifier */
$form18 = "Belgie";
$form19 = "Provinz/Land:";
$v_province = array("Bruxelles", "Brabant Wallon", "Vlaams Brabant", "Hainaut", "Li�ge", "Namur", "Luxembourg", "Antwerpen", "West Vlaanderen", "Oost Vlaanderen", "Limburg", "other");
$a_province = array("Br�ssel", "Brabant Wallon", "Vlaams Brabant", "Hainaut", "L�ttich", "Namur", "Luxembourg", "Antwerpen", "West Vlaanderen", "Oost Vlaanderen", "Limburg", "andere");

$v_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "other");
$a_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Andere");

$form20 = "Das Ausf�llen dieser Angaben ist Ihnen freigestellt. Wir
m�chten Sie aber bitten so viele Angaben wie m�glich zu machen, damit
wir die Intensit�ten so exakt wie m�glich bestimmen k�nnen.";

$t_sit = "In welcher Situation befanden Sie sich w�hrend des Bebens ?";
$o_sit = array("keine Antwort", "im Geb�ude", "im Freien",
    "in einem stehenden Fahrzeug", "in einem bewegten Fahrzeug",
    "sonstiges");

$t_build = "Falls Sie in einem Geb�ude waren, bitte geben Sie 
die Art des Bauwerkes an:";
$o_build = array("kein Geb�ude", "Einfamilienhaus", "Mehrfamilienhaus",
    "B�rogeb�ude/Schule",
    "Mobile Home mit festem Fundament",
    "Wohnwagen ohne Fundament",
    "sonstiges");

$form21 = "Wenn Sie das Stockwerk kennen, bitte nennen Sie es:";
$form22 = "Falls sonstiges, bitte beschreiben:";

$t_sleep = "Haben Sie w�hrend des Bebens geschlafen ?";
$o_sleep = array("Nein", "Ja, und ich habe es verschlafen", "Ja, ich bin aufgewacht");

$form23 = "<b>Haben Sie das Beben gesp�rt ?</b>
(Falls Sie schliefen, sind Sie aufgewacht?)";
$yes = "Ja";
$no = "Nein";
$t_ofelt = "Haben andere Personen in Ihrer N�he das Beben gesp�rt ?";
$o_ofelt = array("keine Antwort / weiss nicht / niemand sonst in der N�he",
    "Niemand sonst hat es gesp�rt",
    "Manche haben es gesp�rt, aber die meisten nicht",
    "Die meisten haben es gep�rt, aber manche nicht",
    "Alle oder fast alle haben es gesp�rt");
	
	
$form24 = "Ihr Empfinden des Erdbebens:";
$t_motion = "Wie w�rden Sie die Bodenbewegungen am besten beschreiben ?";
$o_motion = array("keine Angaben",
    "nicht gesp�rt", "sehr schwach", "schwach",
    "moderat", "stark", "sehr stark");

$form25 = "Wie viele Sekunden etwa dauerte die Ersch�tterung ?";

$t_reaction = "Wie w�rden Sie Ihre Reaktion beschreiben ?";
$o_reaction = array("keine Angaben / kann mich nicht erinnern",
    "Keine Reaktion / nicht gesp�rt",
    "Sehr schwache Reaktion", "Erregt",
    "Leicht ver�ngstigt", "Sehr ver�ngstigt", "Gro�e Angst");

$t_response = "Wie reagierten Sie ? (bitte ausw�hlen)";
$o_response = array("keine Angaben / kann mich nicht erinnern",
    "Habe nicht reagiert",
    "Ging zur T�r",
    "Habe mich geduckt und untergestellt", "Bin nach drau�en gelaufen", "Sonstiges");
$form26 = "Falls sonstiges, bitte beschreiben:";

$t_stand = "Hatten Sie Probleme zu stehen oder zu gehen ?";
$o_stand = array ("keine Angaben / habe es nicht versucht",
    "Nein", "Ja, es war schwer zu stehen",
    "Ja, ich bin gefallen", "Ja, ich wurde vehement zu Boden geworfen");

$form27= "Erdbebeneffekte auf M�bel und Geb�ude :";

$t_sway = "Haben Sie das Schwingen oder die Bewegung von T�ren oder h�ngender Gegenst�nde bemerkt ?";
$o_sway = array ("keine Angaben / habe nicht darauf geachtet",
    "Nein","Ja, leichtes Schwingen","Ja, starkes Schwingen");

$t_creak = "Haben Sie Ger�usche wahrgenommen ?";
$o_creak = array ("keine Angaben / habe nicht darauf geachtet",
        "Nein","Ja, schwache Ger�usche","Ja, laute Ger�usche");

$t_shelf = "Haben Gegenst�nde geklappert, sind umgefallen oder aus Regalen gefallen ?";
$o_shelf = array ("keine Angaben / keine Regale",
            "Nein","Leichtes Klappern","Lautes Klappern",
			"Ein paar sind umgefallen oder heruntergefallen",
			"Viele sind heruntergefallen",
			"Fast alle sind heruntergefallen");

$t_picture = "Wurden Bilder an der Wand verschoben ?";
$o_picture = array("keine Angaben / keine Bilder",
            "Nein","Ja, aber nicht herabgefallen",
			"Ja, manche sind herabgefallen");
			
$t_furniture = "Wurden M�bel oder Einrichtungsgegenst�nde verschoben oder fielen um  ?";
$o_furniture = array("keine Angaben / keine M�bel","Nein","Ja");


$t_heavy_appliance = "Waren schwere Gegenst�nde (K�hlschrank, Herd) betroffen ?";
$o_heavy_appliance = array("keine Angaben / keine schweren Gegenst�nde",
                  "Nein","Ja, Inhalt fiel heraus",
                  "Ja, um einige Zentimeter verschoben",
				  "Ja, um mehr als 30 cm verschoben",
				  "Ja, umgeworfen");

$t_walls = "Wurden frei stehende Mauern oder Z�une besch�digt ?";
$o_walls = array("keine Angaben/keine Mauern",
           "Nein","Ja, manche bekamen Risse",
           "Ja, manche fielen teilweise um",
		   "Ja, manche fielen ganz um");
 
$t_d_text ="Falls Sie drinnen waren, gab es Sch�den am Geb�ude ? Klicken Sie alles Zutreffende an."; 
$o_d_text =array("Keine Sch�den",
                 "Haarrisse in W�nden",
				 "Wenige gr��ere Risse in W�nden",
				 "Viele gr��ere Risse in W�nden",
				 "Deckenverkleidung oder Lampen fielen herab",
				 "Riss in Schornsteinen",
				 "Ein oder mehrere Spr�nge in Fenstern",
				 "Viele Fenster mit Rissen, manche zerbrochen",
				 "Putz fiel von W�nden",
				 "Alter Schornstein zeigt starke Sch�den oder fiel herab",
				 "Neuerer Schornstein zeigt starke Sch�den oder fiel herab",
				 "Au�enw�nde neigten sich oder fielen ganz um",
				 "Risse zwischen Haus und Balkon, Garage oder anderen Anbauten",
	             "Geb�ude wurde vom Fundament verschoben"); 
 
$form28="Falls der Bautentyp (Holz, Ziegelstein, ...) und/oder
      die H�he (Anzahl der Stockwerke) bekannt ist, bitte hier angeben :";
 
 
$form29="Zus�tzliche Kommentare:";
$form30="Wenn Sie weitere Beobachtungen gemacht haben, benutzen Sie bitte die folgende
Textbox. <BR> (Bei Fragen k�nnen Sie uns direkt per E-mail kontaktieren, bitte verwenden 
Sie daf�r NICHT diese Textbox.)";

$form31="Klicken Sie den Sendeknopf, um den Fragebogen abzuschicken :"; 
$form32="Klicken Sie hier, um den ganzen Fragebogen zu l�schen :"; 

$form33="Vielen Dank f�r Ihre Hilfe. Probleme oder Korrekturen bitte an: ";
$form34=" Modifications : ROB - German Version: Erdbebenstation Bensberg 2001/".date("Y")." ";
$form35="Verifizierungscode";

$warn0 = "WARNING ";
$warn1 = "is not an accepted country in this inquiry.";
$warn2 = "Please consult the website for this country.";
$warn3 = "is not a valid e-mail address.";
$warn4 = $form15;
$warn5 = "Not found in our database.";
$warn6 = "Searching for the village : ";
$warn7 = $form14;
$warn8 = "You are not allowed to send two forms for the same event and for the same locality.";
$warn9 = "Postleitzahl nicht gefunden im Datenbank.";
$warn10 = "Verbindung mit Databank nicht gelungen.";
$warn11 = "Gemeinde nicht gefunden im Datenbank.";
$warn12 = "Die �berpr�fungskode ist nicht g�ltig.";
$warn13 = "Geben Sie bitte die ungef�hre Uhr der Ereignis (HH:MM).";
$warn14 = "Geben Sie bitte ihr Postleitzahl.";
$warn15 = "Geben Sie bitte die �berpr�fungskode."; 
$warn16 = "W�hlen Sie bitte ihr Postleitzahl.";

$rep0 = "Macroseismic Inquiry :";
$rep1 = "Thank you for completing this form.";
$rep2 = "We have registered your answer.";
$rep3 = "The locality where you have experienced the earthquake :";
$rep4 = "Your e-mail address :";
$rep5 = "Intensity calculated for your village based on your answers :";
$rep6 = "Close";

$secu1 = "Geben Sie die Zeichen aus dem unten angezeigten Bild ein";
$secu2 = "Bei Buchstaben wird nicht zwischen Gro� und Kleinschreibung unterschieden. ";

$t_noise = "Haben Sie ein Ger�usch geh�rt ?";
$o_noise = array("Nein",
    "Ja, leises und kurzes Ger�usch",
    "Ja, leises und l�ngeres Ger�usch",
    "Ja, lautes und kurzes Ger�usch",
    "Ja, lautes und langes Ger�usch"); 
?>
