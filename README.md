# Parkinsons Freezing of Gait Prediction
Event detection from wearable sensor data



Description: Goal of the Competition

The goal of this competition is to detect freezing of gait (FOG), a debilitating symptom that afflicts many people with Parkinson’s disease. You will develop a machine learning model trained on data collected from a wearable 3D lower back sensor.

Your work will help researchers better understand when and why FOG episodes occur. This will improve the ability of medical professionals to optimally evaluate, monitor, and ultimately, prevent FOG events.

Context:

An estimated 7 to 10 million people around the world have Parkinson’s disease, many of whom suffer from freezing of gait (FOG). During a FOG episode, a patient's feet are “glued” to the ground, preventing them from moving forward despite their attempts. FOG has a profound negative impact on health-related quality of life—people who suffer from FOG are often depressed, have an increased risk of falling, are likelier to be confined to wheelchair use, and have restricted independence.

While researchers have multiple theories to explain when, why, and in whom FOG occurs, there is still no clear understanding of its causes. The ability to objectively and accurately quantify FOG is one of the keys to advancing its understanding and treatment. Collection and analysis of FOG events, such as with your data science skills, could lead to potential treatments.

There are many methods of evaluating FOG, though most involve FOG-provoking protocols. People with FOG are filmed while performing certain tasks that are likely to increase its occurrence. Experts then review the video to score each frame, indicating when FOG occurred. While scoring in this manner is relatively reliable and sensitive, it is extremely time-consuming and requires specific expertise. Another method involves augmenting FOG-provoking testing with wearable devices. With more sensors, the detection of FOG becomes easier, however, compliance and usability may be reduced. Therefore, a combination of these two methods may be the best approach. When combined with machine learning methods, the accuracy of detecting FOG from a lower back accelerometer is relatively high. However, the datasets used to train and test these algorithms have been relatively small and generalizability is limited to date. Furthermore, the emphasis has been on achieving high levels of accuracy, while precision, for example, has largely been ignored.

Competition host, the Center for the Study of Movement, Cognition, and Mobility (CMCM), Neurological Institute, Tel Aviv Sourasky Medical Center, aims to improve the personalized treatment of age-related movement, cognition, and mobility disorders and to alleviate the associated burden. They leverage a combination of clinical, engineering, and neuroscience expertise to: 1) Gain new understandings into the physiologic and pathophysiologic mechanisms that contribute to cognitive and motor function, the factors that influence these functions, and their changes with aging and disease (e.g., Parkinson’s disease, Alzheimer’s). 2) Develop new methods and tools for the early detection and tracking of cognitive and motor decline. A major focus is on using leveraging wearable devices and digital technologies; and 3) Develop and evaluate novel methods for the prevention and treatment of gait, falls, and cognitive function.

Your work will help advance the evaluation, understanding and treatment of FOG, improving the lives of the many people who suffer from this debilitating Parkinson’s disease symptom.

<img width="770" height="429" alt="Screenshot 2025-11-06 at 11 46 48 PM" src="https://github.com/user-attachments/assets/ebbd9e71-6ba7-4c26-8f03-aa7695758bda" />
<img width="815" height="662" alt="Screenshot 2025-11-06 at 11 47 00 PM" src="https://github.com/user-attachments/assets/ba45ef4b-4906-49aa-bca5-8ece74513489" />

Data Sources: https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data
Additional Data Documentation
This page is a supplement to the documentation on the Data page. We give video examples of freezing gait events and describe the collection protocols for the three data sources.

Video Examples of Freezing Gait Events
Parkinsonian Gait Demonstration - Short demonstration of FOG events.
Freezing of gait from The Lancet - Patient showing FOG events during a test.
Multitarget tDCS for Freezing of Gait in Parkinson’s Disease - Example of a FOG provoking test.
Freezing of Gait & Interventions For Freezing Triggers - See 7:20 for descriptions of the event types.
Gait impairments in Parkinson's disease
Cycling for Freezing Gait in Parkinson's Disease
Parkinson's Freezing of Gait - Before and After Exercise
Parkinson's Disease Freezing & Festinating Gait
Parkinson's Disease: Freezing of Gait - Parkinsonism and Related
The tDCS FOG Dataset
Subjects arrived to the clinic for multiple visits as described in Reches T, Dagan M. et al., 2021 and Manor B, Dagan M et al. 2021. At each visit they completed the freezing of gait provoking protocol described by Ziegler et al. 2010 (DOI: 10.1002/mds.22993) both at "off" and "on" anti-parkinsonian medications while wearing 3D accelerometer on their lower back (Opals by APDM Wearable Technologies, Portland, OR, USA. Sampling rate 128Hz). All FoG provoking trials were videotaped and analyzed offline.

Data recordings include a short period (~2-3s) of quiet standing before the start of the test protocol.

The DeFOG Dataset
This project included two visits at the subject's home environment. At each visit, the participants were evaluated at Off and On medication states. During the motor assessment, the subjects wore a 3D accelerometer on their lower back (Ax3 by Axivity) which recorded the data with a sample rate of 100Hz. The acceleration units are provided in [g].

PROTOCOL
The protocol that was performed at each visit:

At visit 1: During Off medication:

4-meter walk test
Timed Up & Go (TUG) - Single task
Timed Up & Go (TUG) – Dual-task (subtracting numbers while performing the TUG test)
Turning task with alternating directions- Single task (performing 4 x 360 degrees turns, each time alternating the rotation direction).
Turning task – Dual-task (same as before, but with additional number subtraction task).
Hotspot Door – A walking trial that involves opening a door, entering another room, turning, and returning to the start point.
Personalized Hotspot - walking through an area in the house that the subject describes as FoG provoking.
During On medication: The protocol is repeated again with the addition of a MiniBest test that is added after the 4 meters walk. See at the end of the protocol elaboration about the MiniBest test.

At visit 2: The same protocol that is described for visit 1 was repeated. In addition, the tasks were also performed with auditory cueing, with the exception of tasks that includes dual task (e.g. "Timed Up & Go (TUG) – Dual task" and "Turning task – Dual task").

MiniBest test includes the following parts:

SIT TO STAND.
RISE TO TOES.
STAND ON ONE LEG.
COMPENSATORY STEPPING CORRECTION- FORWARD.
CSC BACKWARD.
CSC LATERAL.
STANCE (FEET TOGETHER) EYES OPEN, FIRM SURFACE.
STANCE (FEET TOGETHER) EYES CLOSED, FOAM SURFACE.
INCLINE EYES CLOSED (Shoulder width, arms at your side).
DYNAMIC GAIT INDEX:
CHANGE IN GAIT SPEED.
WALK WITH HEAD TURNS – HORIZONTAL.
WALK WITH PIVOT TURNS.
STEP OVER OBSTACLES.
The Daily Living Dataset
The Daily-living contains data from 65 people with PD that were recorded using the same device as in the home FoG-provoking dataset (the DeFOG dataset). The daily-living recordings contain ~one week of unlabeled, continuous recordings from an accelerometer device placed on the lower back of the subjects at 100Hz, during their daily living activity.

The 65 PD subjects are comprised of two groups:

45 people with PD that suffer from FoG. They also underwent a FoG-provoking protocol at their home and that data is provided in the DeFOG data set. In the metadata file, these subjects have NFOG questionnaire score higher than 0.
20 people with PD that do not suffer from FoG. In the metadata file, these subjects have NFOG questionnaire score of 0.

Citation:

Addison Howard, amit salomon, eran gazit, Jeff Hausdorff, Leslie Kirsch, Maggie, Pieter Ginis, Ryan Holbrook, and Yasir F Karim. Parkinson's Freezing of Gait Prediction. https://kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction, 2023. Kaggle.
