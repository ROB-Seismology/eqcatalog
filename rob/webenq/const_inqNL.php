<?php
$inqtitre = "Hebt u de aardbeving gevoeld? Laat het ons weten !";
$intro[1] = "Door de onderstaande vragenlijst in te vullen stelt u ons 
in staat om de grootte van de zone te bepalen waar de aardbeving 
werd gevoeld met daarbijhorend een schatting van de aangerichte 
schade. Uw antwoorden zullen ons ook de mogelijkheid geven de 
wijze te bepalen waarop uw streek kan getroffen worden door 
toekomstige aardbevingen.";

$intro[2] = "De onderzoekers van de Koninklijke Sterrenwacht van België 
zullen de binnenkomende informatie gebruiken voor publicaties, 
waarin op kwalitatieve, kwantitatieve of grafische wijze de 
schade zal beschreven worden. Persoonlijke informatie zal niet 
worden verspreid. Vul de vragenlijst niet in indien u niet akkoord 
gaat met dergelijke informatieverwerking.";

$intro[3] = "<b>Uw postcode is absoluut noodzakelijk</b> om de intensiteit
van de aardbeving in uw woonplaats te bepalen. Alle bijkomende
informatie (naam, e-mail, telefoonnummer en adres) is facultatief,
maar kan bijdragen tot een betere lokale evaluatie van de intensiteit.";

$intro[4] = "Indien u informatie hebt over de manier waarop de aardbeving 
andere plaatsen heeft getroffen, vul dan voor iedere nieuwe 
locatie een andere vragenlijst in.";

$form1 = "VRAGENLIJST OVER DE AARDBEVING VAN:";
$form2 = "Let op: vul het formulier in horende bij de juiste aardbeving.";
$form3 = "VRAGENLIJST OVER EEN NIEUWE AARDBEVING";
$form4 = "Datum en tijdstip van de schok:";
$form5 = "Maand:";
$amois = array("---", "Jan.", "Feb.", "Maart", "Apr.", "Mei", "Juni", "Juli", "Aug.", "Sep.", "Okt.", "Nov.", "Dec.");
$form6 = "Dag:";
$form7 = "Jaar:";
$form8[0] = "Uur(UU:MM):";
$form8[1] = "'s Morgens";
$form8[2] = "'s Namiddags";
$form9 = "Naam:";
$form10 = "Telefoonnummer:";
$form11 = "Waar bevond u zich op het ogenblik van de aardbeving ?";
$form12 = "Vul indien mogelijk alle volgende vakken in, aangezien het gaat om
  een nog niet geïnventariseerde aardbeving. Deze informatie moet ons
  toelaten om de plaats van de aardbeving nauwkeuriger te bepalen.<br>";
$form13 = "Straat,<br> Adres:";
$form14 = "Gemeente:";
$form15 = "Postcode:";
$form16 = "(VEREIST !)";

$form17 = "Land:";
$v_country = array("BE","FR","NL","DE","LU","GB","??");
$a_country = array("België","Frankrijk","Nederland","Duitsland","Luxemburg","Engeland","Ander land");
//$v_country = array("BE", "LU", "??");
//$a_country = array("België", "Luxemburg", "Ander land");

$form19 = "Provincie:";
$v_province = array("Antwerpen", "Bruxelles", "Limburg", "Oost Vlaanderen", "Vlaams Brabant", "West Vlaanderen", "Brabant Wallon", "Hainaut", "Liège", "Luxembourg", "Namur", "other");
$a_province = array("Antwerpen", "Brussels Hoofdstedelijk Gewest", "Limburg", "Oost-Vlaanderen", "Vlaams-Brabant", "West-Vlaanderen", "Brabant Wallon", "Hainaut", "Liège", "Luxembourg", "Namur", "Andere provincie");

$v_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "other");
$a_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Ander");

$form20 = "Het antwoord op alle vragen is facultatief, maar we hopen 
    op een zo volledig mogelijke antwoordenlijst, die ons de 
    mogelijkheid moet geven om de intensiteit van de aardbeving op de 
    aangegeven plaats zo nauwkeurig mogelijk te bepalen.";

$t_sit = "Waar bevond u zich op het ogenblik van de aardbeving?";
$o_sit = array("Geen antwoord", "Binnen", "Buiten",
    "In een stilstaand voertuig", "In een rijdend voertuig",
    "Ergens anders");

$t_build = "Indien u zich binnen bevond, <br>
    beschrijf dan het type van het gebouw:";
$o_build = array("Geen gebouw", "Eengezinswoning", "Appartementsgebouw",
    "Kantoor, fabriek of school",
    "Mobilhome met vaste funderingen",
    "Caravan zonder funderingen, tent",
    "Andere constructie");

$form21 = "Op welke verdieping bevond u zich?";
$form22 = "Indien u zich in een andere constructie bevond, beschrijf a.u.b:";
$t_sleep = "Sliep u tijdens de aardbeving?";
$o_sleep = array("Neen", "Ja, ik ben niet wakker geworden", "Ja, ik ben wakker geworden");

$form23 = "Hebt u de aardbeving gevoeld?<br>(Indien u sliep, werd u  gewekt door de aardbeving?)";
$yes = "Ja";
$no = "Neen";

$t_ofelt = "Hebben andere personen in uw omgeving de aardbeving gevoeld?";
$o_ofelt = array("Geen antwoord / Ik weet het niet / Niemand in de omgeving",
    "Niemand anders heeft iets gevoeld",
    "Sommigen hebben iets gevoeld maar niet iedereen",
    "De meesten hebben iets gevoeld op enkele uitzonderingen na",
    "Iedereen heeft het gevoeld");
	
$form24 = "Hoe hebt u de aardbeving ervaren?";

$t_motion = "Hoe zou u de schok beschrijven?";
$o_motion = array("Geen beschrijving","Niet gevoeld", "Zwak", "Licht",
    "Matig", "Zwaar", "Zeer zwaar");

$form25 = "Hoeveel seconden ongeveer was de schok voelbaar ?";

$t_reaction = "Hoe zou u het best uw reactie omschrijven?";
$o_reaction = array("Geen antwoord / Ik herinner me het niet",
    "Geen reactie / Niet gevoeld",
    "Lichte reactie", "Opwinding",
    "Licht opgeschrikt", "Zwaar opgeschrikt", "Ik was in paniek");

$t_response = "Welke actie hebt u ondernomen? (Kies een antwoord)";
$o_response = array("Geen antwoord / Ik herinner me het niet",
    "Geen bijzondere reactie",
    "Ik heb me naar de uitgang begeven",
    "Ik heb dekking gezocht", "Ik heb onmiddellijk de plaats verlaten", "Andere reactie");
$form26 = " Indien u een andere reactie had, beschrijf a.u.b. :";

$t_stand = "Was het moeilijk om rechtop te blijven of om te gaan?";
$o_stand = array ("Geen antwoord / Ik heb niet geprobeerd",
     "Neen", "Ja, het was moeilijk om mijn evenwicht te houden",
    "Ja, ik ben gevallen", "Ja, ik werd op de grond geworpen");

$form27= "Invloed van de aardbeving op het meubilair en de gebouwen";

$t_sway = "Hebt u het slingeren of zwaaien waargenomen van deuren of opgehangen voorwerpen?";
$o_sway = array ("Geen antwoord / Ik heb het niet gecontroleerd",
        "Neen", "Ja, lichte schommelingen", "Ja, sterk heen- en weergaande bewegingen");

$t_creak = "Hebt u het horen kraken of hebt u nog andere geluiden waargenomen ?";
$o_creak = array ("Geen antwoord / Ik heb er geen aandacht aan geschonken",
                   "Neen", "Ja, ik heb lichte geluiden waargenomen",
                   "Ja, ik heb sterke geluiden waargenomen");

$t_shelf = "Zijn er voorwerpen tegen mekaar gebotst, zijn ze omgekanteld of
       van de rekken gevallen?";
$o_shelf = array ("Geen antwoord / Geen rekken",
            "Neen", "Lichte geluiden van trillende en botsende voorwerpen",
            "Tamelijk sterke geluiden van trillende en botsende voorwerpen",
			"Sommige voorwerpen zijn gekanteld of gevallen",
			"Vele voorwerpen zijn gevallen",
			"Bijna alle voorwerpen zijn gevallen");

$t_picture = "Hebben kaders of schilderijen aan de wand bewogen ?";
$o_picture = array("Geen antwoord / Geen kaders",
            "Neen", "Ja, maar ze zijn niet gevallen",
			"Ja, sommige zijn gevallen");
			
$t_furniture = "Werden er meubelen verplaatst?";
$o_furniture = array("Geen antwoord / Geen meubelen", "Neen", "Ja");


$t_heavy_appliance = "Had de aardbeving ook een effect op zware meubelen zoals bv. een koelkast?";
$o_heavy_appliance = array("Geen antwoord / Geen zware meubelen",
                  "Neen", "Ja, een deel van de inhoud is omgevallen",
                  "Ja, de meubelen werden enkele centimeters verplaatst",
				  "Ja, de meubelen werden over minstens 30 centimeter verplaatst",
				  "Ja, de meubelen zijn omgevallen");

$t_walls = "Werden alleenstaande muren of omheiningen beschadigd?";
$o_walls = array("Geen antwoord / Geen muren",
           "Neen", "Ja, sommige muren vertonen barsten",
           "Ja, sommige muren zijn gedeeltelijk omgevallen",
		   "Ja, sommige muren zijn volledig ingestort");

				  
 
$t_d_text ="Indien u zich op het moment van de aardbeving binnen bevond,
    hoe heeft het gebouw de schok doorstaan? Kruis alles aan wat van toepassing is:"; 
$o_d_text =array("Geen schade",
                 "Lichte barsten in de muren",
				 "Enkele tamelijk grote barsten in de muren",
				 "Talrijke grote barsten in de muren",
				 "Plafondtegels of lichtarmaturen zijn gevallen",
				 "Er zijn barsten in de schoorsteen",
				 "Eén of enkele vensters zijn gebarsten",
				 "Talrijke vensters zijn gebarsten en enkele zijn gewoon verbrijzeld",
				 "Stukken metselwerk zijn losgekomen of gevallen",
				 "Een oude schoorsteen is zwaar beschadigd of gevallen",
				 "Een recente schoorsteen is zwaar beschadigd of gevallen",
				 "Buitenmuren hellen over of zijn volledig ingestort",
				 "Er zijn barsten ontstaan tussen de woning en de aangebouwde of
    overhangende delen van de woning: veranda's, balkons, ...",
	             "Het gebouw werd van zijn funderingen losgerukt en verplaatst"); 
 
 
$form28="Kan u het materiaal aangeven waarin het gebouw is opgetrokken
    (hout, baksteen, natuursteen, ...) alsook de hoogte en/of het aantal verdiepingen?";
 
 
$form29="Opmerkingen";
$form30="In het tekstveld hieronder kunt u eventueel bijkomende waarnemingen invullen: <BR> (Hier geen vragen stellen. Gebruik de e_mail.)";

$form31="Om uw formulier op te sturen, drukt u op deze toets :"; 
$form32="Druk op deze toets om alles te wissen en te herbeginnen"; 

$form33="We danken u voor uw medewerking. Voor eventuele vragen of suggesties
kunt u een e-mail sturen naar: ";
$form34="Vertaling en aanpassingen: Koninklijke Sterrenwacht van België - 2001/".date("Y")." ";
$form35="Verification Code";

$warn0 = "OPGELET ";
$warn1 = "is een land dat niet wordt geaccepteerd voor deze enquête.";
$warn2 = "Gelieve de website voor dit land te raadplegen.";
$warn3 = "is geen geldig email-adres.";
$warn4 = $form15;
$warn5 = "Niet gevonden in onze gegevensbank.";
$warn6 = "Opzoeken van de stad of gemeente : ";
$warn7 = $form14;
$warn8 = "U mag geen twee verschillende formulieren opsturen voor dezelfde aardbeving en dezelfde locatie.";
$warn9 = "Postcode niet gevonden in database.";
$warn10 = "Verbinding met database mislukt.";
$warn11 = "Gemeente of stad niet gevonden in database.";
$warn12 = "De verificatiecode is niet geldig.";
$warn13 = "Gelieve het tijdstip van de gebeurtenis bij benadering op te geven (HH:MM).";
$warn14 = "Gelieve uw postcode in te vullen.";
$warn15 = "Gelieve de verificatiecode in te vullen."; 
$warn16 = "Gelieve uw postcode in te vullen."; //?

$rep0 = "Macroseismische Enquète :";
$rep1 = "We danken u voor uw medewerking.";
$rep2 = "Wij hebben uw antwoord geregistreerd.";
$rep3 = "De plaats waar u de aardbeving gevoeld hebt :";
$rep4 = "Uw email-adres :";
$rep5 = "Intensiteit berekend voor uw stad of gemeente op basis van uw antwoorden :";
$rep6 = "Sluiten";

$secu1 = "Geef de tekens op die in de onderstaande afbeelding worden weergegeven";
$secu2 = "Letters zijn niet hoofdlettergevoelig";

$t_noise = "Hebt u een geluid gehoord ?";
$o_noise = array("Neen",
    "Ja, licht en kortstondig",
    "Ja, licht en langdurig",
    "Ja, luid en kortstondig",
    "Ja, luid en langdurig"); 
?>