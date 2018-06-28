<?php
$inqtitre = "Haben Sie ein Erdbeben gespürt ?  Bitte geben Sie Ihren Bericht hier ein :";
$intro[1] = "Sie können uns helfen die Ausdehnung der Erschütterungen und 
Schäden von Erdbeben zu erfassen. Sie helfen damit, 
Einzelheiten der Wirkung zukünftiger Beben in Ihrer Region abzuschätzen und die Erdbebenvorsorge zu verbessern.";

$intro[2] = "KSB und BNS Wissenschaftler können die Informationen, die Sie in diesen
Fragebogen eingeben, benutzen, um in Ihren Publikationen die Wirkung
von Erdbeben qualitativ, quantitativ oder graphisch zu beschreiben.
Ihre persönlichen Daten werden nicht weiterverwendet. Wenn Sie dieser
Verwendung Ihrer Angaben nicht zustimmen, dann füllen Sie den Fragebogen
bitte nicht aus.";

$intro[3] = "<b>Ihre POSTLEITZAHL ist ERFORDERLICH </b>, um die Intensität in
Ihrer Gegend festlegen zu können. Alle anderen Angaben (Name, e-mail, 
Telefonnummer und Ortsangabe) sind optional, können aber für 
Detailuntersuchungen wichtig sein.";

$intro[4] = "Wenn Sie Informationen von mehr als einem Ort haben, so füllen Sie bitte für jeden Ort ein eigenes Formular aus.";

$form1 = "FRAGEBOGEN FÜR:";
$form2 = "Vergewissern sie sich bitte das dem Ereignis entsprechende Formular auszufüllen.";
$form3 = "FRAGEBOGEN FÜR EIN NEUES ERDBEBEN";
$form4 = "Datum und Zeit des Erdbebens (ungefähr):";
$form5 = "Monat:";
$amois = array("---", "Jan.", "Feb.", "Mär.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.", "Sep.", "Okt.", "Nov.", "Dez.");
$form6 = "Tag:";
$form7 = "Jahr:";
$form8[0] = "Uhrzeit (HH:MM):";
$form8[1] = "AM";
$form8[2] = "PM";
$form9 = "Name:";
$form10 = "Telefon:";
$form11 = "Ihr Aufenthaltsort zu dem Zeitpunkt, als Sie das Beben wahrgenommen haben ?";
$form12 = "Da Sie das Formular für ein noch nicht aufgeführtes Erdbeben ausfüllen, 
bitten wir Sie die folgenden Informationen komplett anzugeben. Dies wird 
uns helfen das Erdbeben genauestens zu lokalisieren.";
$form13 = "Straße,<BR>Anschrift:";
$form14 = "Stadt:";
$form15 = "PLZ:";
$form16 = "(ERFORDERLICH!)";

$form17 = "Land:";
$v_country = array("BE","FR","NL","DE","LU","GB","??");
$a_country = array("Belgien","Frankreich","Niederlande","Deutschland","Luxemburg","Großbritannien","Andere");
//$v_country = array("BE", "LU", "??");
//$a_country = array("Belgïe", "Luxemburg", "Andere");

/* a vérifier */
$form18 = "Belgie";
$form19 = "Provinz/Land:";
$v_province = array("Bruxelles", "Brabant Wallon", "Vlaams Brabant", "Hainaut", "Liège", "Namur", "Luxembourg", "Antwerpen", "West Vlaanderen", "Oost Vlaanderen", "Limburg", "other");
$a_province = array("Brüssel", "Brabant Wallon", "Vlaams Brabant", "Hainaut", "Lüttich", "Namur", "Luxembourg", "Antwerpen", "West Vlaanderen", "Oost Vlaanderen", "Limburg", "andere");

$v_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "other");
$a_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Andere");

$form20 = "Das Ausfüllen dieser Angaben ist Ihnen freigestellt. Wir
möchten Sie aber bitten so viele Angaben wie möglich zu machen, damit
wir die Intensitäten so exakt wie möglich bestimmen können.";

$t_sit = "In welcher Situation befanden Sie sich während des Bebens ?";
$o_sit = array("keine Antwort", "im Gebäude", "im Freien",
    "in einem stehenden Fahrzeug", "in einem bewegten Fahrzeug",
    "sonstiges");

$t_build = "Falls Sie in einem Gebäude waren, bitte geben Sie 
die Art des Bauwerkes an:";
$o_build = array("kein Gebäude", "Einfamilienhaus", "Mehrfamilienhaus",
    "Bürogebäude/Schule",
    "Mobile Home mit festem Fundament",
    "Wohnwagen ohne Fundament",
    "sonstiges");

$form21 = "Wenn Sie das Stockwerk kennen, bitte nennen Sie es:";
$form22 = "Falls sonstiges, bitte beschreiben:";

$t_sleep = "Haben Sie während des Bebens geschlafen ?";
$o_sleep = array("Nein", "Ja, und ich habe es verschlafen", "Ja, ich bin aufgewacht");

$form23 = "<b>Haben Sie das Beben gespürt ?</b>
(Falls Sie schliefen, sind Sie aufgewacht?)";
$yes = "Ja";
$no = "Nein";
$t_ofelt = "Haben andere Personen in Ihrer Nähe das Beben gespürt ?";
$o_ofelt = array("keine Antwort / weiss nicht / niemand sonst in der Nähe",
    "Niemand sonst hat es gespürt",
    "Manche haben es gespürt, aber die meisten nicht",
    "Die meisten haben es gepürt, aber manche nicht",
    "Alle oder fast alle haben es gespürt");
	
	
$form24 = "Ihr Empfinden des Erdbebens:";
$t_motion = "Wie würden Sie die Bodenbewegungen am besten beschreiben ?";
$o_motion = array("keine Angaben",
    "nicht gespürt", "sehr schwach", "schwach",
    "moderat", "stark", "sehr stark");

$form25 = "Wie viele Sekunden etwa dauerte die Erschütterung ?";

$t_reaction = "Wie würden Sie Ihre Reaktion beschreiben ?";
$o_reaction = array("keine Angaben / kann mich nicht erinnern",
    "Keine Reaktion / nicht gespürt",
    "Sehr schwache Reaktion", "Erregt",
    "Leicht verängstigt", "Sehr verängstigt", "Große Angst");

$t_response = "Wie reagierten Sie ? (bitte auswählen)";
$o_response = array("keine Angaben / kann mich nicht erinnern",
    "Habe nicht reagiert",
    "Ging zur Tür",
    "Habe mich geduckt und untergestellt", "Bin nach draußen gelaufen", "Sonstiges");
$form26 = "Falls sonstiges, bitte beschreiben:";

$t_stand = "Hatten Sie Probleme zu stehen oder zu gehen ?";
$o_stand = array ("keine Angaben / habe es nicht versucht",
    "Nein", "Ja, es war schwer zu stehen",
    "Ja, ich bin gefallen", "Ja, ich wurde vehement zu Boden geworfen");

$form27= "Erdbebeneffekte auf Möbel und Gebäude :";

$t_sway = "Haben Sie das Schwingen oder die Bewegung von Türen oder hängender Gegenstände bemerkt ?";
$o_sway = array ("keine Angaben / habe nicht darauf geachtet",
    "Nein","Ja, leichtes Schwingen","Ja, starkes Schwingen");

$t_creak = "Haben Sie Geräusche wahrgenommen ?";
$o_creak = array ("keine Angaben / habe nicht darauf geachtet",
        "Nein","Ja, schwache Geräusche","Ja, laute Geräusche");

$t_shelf = "Haben Gegenstände geklappert, sind umgefallen oder aus Regalen gefallen ?";
$o_shelf = array ("keine Angaben / keine Regale",
            "Nein","Leichtes Klappern","Lautes Klappern",
			"Ein paar sind umgefallen oder heruntergefallen",
			"Viele sind heruntergefallen",
			"Fast alle sind heruntergefallen");

$t_picture = "Wurden Bilder an der Wand verschoben ?";
$o_picture = array("keine Angaben / keine Bilder",
            "Nein","Ja, aber nicht herabgefallen",
			"Ja, manche sind herabgefallen");
			
$t_furniture = "Wurden Möbel oder Einrichtungsgegenstände verschoben oder fielen um  ?";
$o_furniture = array("keine Angaben / keine Möbel","Nein","Ja");


$t_heavy_appliance = "Waren schwere Gegenstände (Kühlschrank, Herd) betroffen ?";
$o_heavy_appliance = array("keine Angaben / keine schweren Gegenstände",
                  "Nein","Ja, Inhalt fiel heraus",
                  "Ja, um einige Zentimeter verschoben",
				  "Ja, um mehr als 30 cm verschoben",
				  "Ja, umgeworfen");

$t_walls = "Wurden frei stehende Mauern oder Zäune beschädigt ?";
$o_walls = array("keine Angaben/keine Mauern",
           "Nein","Ja, manche bekamen Risse",
           "Ja, manche fielen teilweise um",
		   "Ja, manche fielen ganz um");
 
$t_d_text ="Falls Sie drinnen waren, gab es Schäden am Gebäude ? Klicken Sie alles Zutreffende an."; 
$o_d_text =array("Keine Schäden",
                 "Haarrisse in Wänden",
				 "Wenige größere Risse in Wänden",
				 "Viele größere Risse in Wänden",
				 "Deckenverkleidung oder Lampen fielen herab",
				 "Riss in Schornsteinen",
				 "Ein oder mehrere Sprünge in Fenstern",
				 "Viele Fenster mit Rissen, manche zerbrochen",
				 "Putz fiel von Wänden",
				 "Alter Schornstein zeigt starke Schäden oder fiel herab",
				 "Neuerer Schornstein zeigt starke Schäden oder fiel herab",
				 "Außenwände neigten sich oder fielen ganz um",
				 "Risse zwischen Haus und Balkon, Garage oder anderen Anbauten",
	             "Gebäude wurde vom Fundament verschoben"); 
 
$form28="Falls der Bautentyp (Holz, Ziegelstein, ...) und/oder
      die Höhe (Anzahl der Stockwerke) bekannt ist, bitte hier angeben :";
 
 
$form29="Zusätzliche Kommentare:";
$form30="Wenn Sie weitere Beobachtungen gemacht haben, benutzen Sie bitte die folgende
Textbox. <BR> (Bei Fragen können Sie uns direkt per E-mail kontaktieren, bitte verwenden 
Sie dafür NICHT diese Textbox.)";

$form31="Klicken Sie den Sendeknopf, um den Fragebogen abzuschicken :"; 
$form32="Klicken Sie hier, um den ganzen Fragebogen zu löschen :"; 

$form33="Vielen Dank für Ihre Hilfe. Probleme oder Korrekturen bitte an: ";
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
$warn12 = "Die Überprüfungskode ist nicht gültig.";
$warn13 = "Geben Sie bitte die ungefähre Uhr der Ereignis (HH:MM).";
$warn14 = "Geben Sie bitte ihr Postleitzahl.";
$warn15 = "Geben Sie bitte die Überprüfungskode."; 
$warn16 = "Wählen Sie bitte ihr Postleitzahl.";

$rep0 = "Macroseismic Inquiry :";
$rep1 = "Thank you for completing this form.";
$rep2 = "We have registered your answer.";
$rep3 = "The locality where you have experienced the earthquake :";
$rep4 = "Your e-mail address :";
$rep5 = "Intensity calculated for your village based on your answers :";
$rep6 = "Close";

$secu1 = "Geben Sie die Zeichen aus dem unten angezeigten Bild ein";
$secu2 = "Bei Buchstaben wird nicht zwischen Groß und Kleinschreibung unterschieden. ";

$t_noise = "Haben Sie ein Geräusch gehört ?";
$o_noise = array("Nein",
    "Ja, leises und kurzes Geräusch",
    "Ja, leises und längeres Geräusch",
    "Ja, lautes und kurzes Geräusch",
    "Ja, lautes und langes Geräusch"); 
?>
