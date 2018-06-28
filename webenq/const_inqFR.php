<?php
$inqtitre = "Avez-vous ressenti le tremblement de Terre ? Informez-nous !";
$intro[1] = "Vous pouvez nous aider en nous permettant de définir
l'étendue de la zone de perception et de dommages des
tremblements de terre en Belgique et vous pouvez nous fournir des
détails spécifiques, qui nous permettront de
prévoir la façon dont votre région sera
affectée par de futurs tremblements de terre.";

$intro[2] = "Les scientifiques de l'Observatoire Royal de Belgique
utiliseront ces informations pour des publications futures et pour
décrire de façon qualitative, quantitative ou
graphique les dommages observés. Les informations
personnelles ne seront pas diffusées. Si vous vous opposez
à un tel usage des informations que vous nous fournissez,
ne remplissez pas ce questionnaire.";

$intro[3] = "<b>Votre Code postal est nécessaire !</b> pour pouvoir
déterminer l'intensité du tremblement de terre dans
votre localité. Toutes les autres données(nom, e-mail,
téléphone et adresse) sont facultatives, mais
pourraient s'avérer importantes pour une évaluation
plus locale de l'intensité.";

$intro[4] = "Si vous avez des constatations sur la manière dont le
tremblement de terre a affecté d'autres lieux que celui
où vous vous trouviez, remplissez un autre questionnaire pour
chaque nouvelle localisation.";

$form1 = "QUESTIONNAIRE POUR LE TREMBLEMENT DE TERRE SUIVANT:";
$form2 = "Assurez-vous de remplir le formulaire correspondant au bon événement.";
$form3 = "QUESTIONNAIRE POUR UN NOUVEL ÉVÉNEMENT";
$form4 = "Date et heure du tremblement de terre (approximativement):";
$form5 = "Mois:";
$amois = array("---", "Jan.", "Fev.", "Mar.", "Avr.", "Mai", "Juin", "Juil.", "Aout", "Sep.", "Oct.", "Nov.", "Dec.");
$form6 = "Jour:";
$form7 = "Année:";
$form8[0] = "Heure(HH:MM):";
$form8[1] = "Matin";
$form8[2] = "Après-Midi";
$form9 = "Nom:";
$form10 = "Téléphone:";
$form11 = "Où étiez-vous <i>au moment</i> du tremblement de terre ?";
$form12 = "Comme vous renvoyez ce formulaire pour un événement non encore
  répertorié, <b>remplissez, si possible l'ensemble des cases qui suivent.</b> Ceci nous
  permettra de le localiser avec précision.<br>";
$form13 = "Rue,<br>Adresse:";
$form14 = "Ville:";
$form15 = "Code Postal:";
$form16 = "(REQUIS!)";

$form17 = "Pays:";
$form18 = "Autre";

$v_country = array("BE", "FR", "NL", "DE", "LU", "GB", "??");
$a_country = array("Belgique", "France", "Pays Bas", "Allemagne", "GD du Luxembourg", "Angleterre", "Autre");
//$v_country = array("BE", "LU", "??");
//$a_country = array("Belgique", "GD du Luxembourg", "Autre");

$form19 = "Province:";
$v_province = array("Bruxelles", "Brabant Wallon", "Vlaams Brabant", "Hainaut", "Liège", "Namur", "Luxembourg", "Antwerpen", "West Vlaanderen", "Oost Vlaanderen", "Limburg", "other");
$a_province = array("Bruxelles", "Brabant Wallon", "Vlaams Brabant", "Hainaut", "Liège", "Namur", "Luxembourg", "Antwerpen", "West Vlaanderen", "Oost Vlaanderen", "Limburg", "Autre");

$v_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "other");
$a_province_de = array("Hessen", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Autre");

$form20 = "La réponse à toutes les questions est facultative, mais nous vous encourageons 
        à répondre au maximum de questions possible, afin de pouvoir estimer de la
        façon la plus précise possible l'intensité du tremblement de terre.";

$t_sit = "Où étiez-vous durant le tremblement de terre ?";
$o_sit = array("Pas de réponse", "À l'intérieur", "À l'extérieur",
    "Dans un véhicule à l'arrêt", "Dans un véhicule en mouvement",
    "Autre");

$t_build = "Si vous étiez à l'intérieur, <br>
    précisez type de construction ou de structure :";
$o_build = array("Pas une construction", "Maison unifamiliale", "Immeuble à appartements",
    "Bâtiment à usage professionnel/École",
    "Mobile Home avec fondations permanentes",
    "Caravane ou véhicule récréatif SANS Fondation",
    "Autre");

$form21 = "Si vous connaissez l'étage où vous vous trouviez, précisez s.v.p. :";
$form22 = "Si autre, précisez s.v.p. :";
$t_sleep = "Dormiez-vous lors du tremblement de terre ?";
$o_sleep = array("Non", "Oui, et je ne me suis pas réveillé", "Oui, mais je me suis réveillé");

$form23 = "Avez-vous ressenti le tremblement de terre ?
    (Si vous dormiez, le tremblement de terre vous a-t-il
    réveillé?)";
$yes = "Oui";
$no = "Non";

$t_ofelt = "Est-ce que d'autres personnes à proximité l'ont ressenti ?";
$o_ofelt = array("Pas de réponse / Je ne sais pas / Personne à proximité",
    "Personne d'autre ne l'a ressenti",
    "Certains l'ont ressenti, mais pas la plupart",
    "La plupart l'a ressenti, mais pas certains",
    "Tout le monde ou presque l'a ressenti");

$form24 = "Votre expérience du tremblement de terre :";

$t_motion = "Comment décririez-vous la secousse ?";
$o_motion = array("Pas de description", "Pas ressenti", "Faible", "Légère",
    "Modérée", "Forte", "Violente");

$form25 = "Combien de secondes, approximativement, a duré la secousse ?";

$t_reaction = "comment décririez-vous le mieux votre réaction ?";
$o_reaction = array("Pas de réponse / Je ne me souviens pas",
    "Pas de réaction / Pas ressenti",
    "Réaction très petite", "Excitation",
    "Légèrement effrayé", "Très effrayé", "Panique");

$t_response = "Qu'avez-vous fait ? (Sélectionnez une réponse.)";
$o_response = array("Pas de réponse / Je ne me souviens pas",
    "Pas d'action particulière",
    "Je me suis déplacé vers une porte",
    "Je me suis protégé", "Je suis sorti précipitamment", "Autre");
$form26 = " Si autre, précisez s.v.p. :";

$t_stand = "Etait-il difficile de rester debout ou de marcher ?";
$o_stand = array ("Pas de réponse / Je n'ai pas essayé",
    "Non", "Oui, équilibre difficile",
    "Oui, je suis tombé", "Oui, j'ai été plaqué au sol");

$form27 = "Effets du tremblement de terre sur le mobilier et les constructions:";

$t_sway = "Avez vous remarqué l'oscillation ou le mouvement de portes ou d'objets suspendus ?";
$o_sway = array ("Pas de réponse / Je n'ai pas regardé",
    "Non", "Oui, balancement léger", "Oui, balancement violent");

$t_creak = "Avez-vous entendu des craquements ou d'autres bruits ?";
$o_creak = array ("Pas de réponse / Je n'ai pas fait attention",
    "Non", "Oui, bruit léger", "Oui, bruit fort");

$t_shelf = "Des objets se sont-ils entrechoqués, se sont-ils renversés
			ou sont-ils tombés d'une étagère ?";
$o_shelf = array ("Pas de réponse / Pas d'étagère",
    "Non", "Entrechocs légers", "Entrechocs forts",
    "Certains objets se sont renversés ou sont tombés",
    "Beaucoup d'objets sont tombés",
    "Presque tous les objets sont tombés");

$t_picture = "Les cadres ont-ils bougé ?";
$o_picture = array("Pas de réponse / Pas de cadres",
    "Non", "Oui, mais ne sont pas tombés",
    "Oui, et certains sont tombés");

$t_furniture = "Des meubles ont-ils été déplacés ?";
$o_furniture = array("Pas de réponse / Pas de meubles", "Non", "Oui");

$t_heavy_appliance = "Des meubles lourds ont-ils été affectés (p.e. : réfrigérateur) ?";
$o_heavy_appliance = array("Pas de réponse / Pas de meubles lourds",
    "Non", "Oui, une partie du contenu est tombée",
    "Oui, déplacé de qq cm",
    "Oui, déplacé de plus de 30 cm",
    "Oui, renversé");

$t_walls = "Des murs isolés ou des clôtures ont-ils été endommagés ?";
$o_walls = array("Pas de réponse / Pas de murs",
    "Non", "Oui, certains sont fissurés",
    "Oui, certains sont tombés partiellement",
    "Oui, certains sont complètement écroulés");

$t_d_text = "Si vous étiez à l'intérieur, la construction 
    a-t-elle subi des dommages ? Cochez tout ce qui s'applique.";
$o_d_text = array("Pas de dommages",
    "Fissures étroites dans les murs",
    "Quelques larges fissures dans les murs",
    "De nombreuses fissures larges dans les murs",
    "Plafond, tuiles ou accessoires légers tombés",
    "Fissures dans la cheminée",
    "Une ou plusieurs fêlures dans les fenêtres",
    "De nombreuses fenêtres fêlées ou certaines brisées",
    "Morceaux de maçonnerie tombés des murs",
    "Vieille Cheminée : dommages majeurs ou chute",
    "Cheminée récente : dommages majeurs ou chute",
    "Murs extérieurs inclinés ou complètement écroulés",
    "Séparation du porche, de balcons ou d'autres parties annexes
    accolées au bâtiment",
    "Construction déplacée de sa fondation");

$form28 = "Si vous connaissez le type de construction (bois,
    brique, ...) et / ou la hauteur (ou le nombre d'étages)
    du bâtiment pouvez-vous l'indiquer :";

$form29 = "Commentaires :";
$form30 = "Si vous avez des observations complémentaires, utilisez la fenêtre ci-dessous.<BR> (Ne posez pas de questions ici. Utilisez le mail)";

$form31 = "Pour envoyer votre formulaire, poussez sur le bouton :";
$form32 = "Poussez sur ce bouton pour tout effacer et recommencer :";

$form33 = "Merci d'avoir rempli ce questionnaire. Signalez les problèmes
ou corrections par e-mail à ";
$form34 = " Traduction et modifications : Observatoire Royal de Belgique - 2000/".date("Y")." ";
$form35="Code de vérification";

$warn0 = "ATTENTION ";
$warn1 = "n'est pas un pays accepté.";
$warn2 = "Veuillez utiliser votre site national.";
$warn3 = "n'est pas une adresse e-mail valide.";
$warn4 = $form15;
$warn5 = "Non trouvé dans la base de données";
$warn6 = " Recherche de la ville : ";
$warn7 = $form14;
$warn8 = "Vous ne pouvez pas envoyer deux formulaires pour le même évènement et la même localisation.";
$warn9 = "Code postal non trouvé dans la base de données.";
$warn10 = "Connection impossible à la base de données.";
$warn11 = "Ville non trouvée dans la base de données.";
$warn12 = "Le code de vérification n'est pas valide.";
$warn13 = "Veuillez encoder l'heure approximative de l'événement (HH:MM).";
$warn14 = "Veuillez encoder votre code postal.";
$warn15 = "Veuillez encoder le code de vérification.";
$warn16 = "Veuillez sélectionner votre code postal.";

$rep0 = "Enquète Macroséismique :";
$rep1 = "Merci d'avoir rempli ce questionnaire.";
$rep2 = "Nous avons bien enregistré votre réponse.";
$rep3 = "Le lieu où vous avez ressenti le tremblement de terre :";
$rep4 = "Votre adresse e-mail :";
$rep5 = "Intensité calculée dans votre commune avec vos données :";
$rep6 = "Fermer";

$secu1 = "Entrez les caractères figurant dans l'image ci-dessous";
$secu2 = "La casse n'est pas prise en compte.";

$t_noise = "Avez-vous entendu un bruit ?";
$o_noise = array("Non",
    "Oui, bruit léger et bref",
    "Oui, bruit léger et prolongé",
    "Oui, bruit fort et bref",
    "Oui, bruit fort et prolongé");
?>
