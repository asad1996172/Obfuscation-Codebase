"""
This module shows the usage example for writeprints static feature set
"""
import writeprintsStatic as ws
from JstyloFeatureExtractor import jstyloFeaturesExtractor
import sys

if __name__=="__main__":

	corpus_path = sys.argv[1]
	jfe = jstyloFeaturesExtractor(corpus_path)
	dummy_text = """  The workers could never get out of the cycle because many of them were foreigners and could not speak English let alone get a job somewhere else. To make the point even more, they had no way of moving or the time to look for a different job because their pay was so low that they had just enough money left over each week to eat. I agree with Jurgis' decision to run away because of the great despair he saw all around him in Packingtown, but I believe he should have given more thought to the people he was leaving behind and the immense burden he was putting on them.

	Millions of people die from cancer, disease, and car wrecks each year. These horrible tragedies deserve attention, but murders are often seen with a higher level of emotion. Accidental and unpreventable deaths are always sad, but do not seem as vicious and stinging as murder and violent crime. Maybe the horror of murder is based on the atrocious idea of one human taking the life of another or the pain of forever having to wrestle with the question of "why?" All deaths that are not a result of natural causes should be addressed, but violent crime and murder deserve a greater amount of attention. America needs to figure out how it can combat the nearly 350,000 murders, robberies, and aggravated assaults involving guns each year . Every American wants a safe community to raise a family and live a happy, peaceful life. No one wants to see violence and murder in the streets. Gun crime remains an important issue that must be dealt with in order to maintain unity and safety in our society.
		Guns are an important part of American culture. Everyone owned guns during the formation of the country, and men used privately owned weapons to protect their families and countrymen. During the 1930s gun control became a controversial issue as the government began to pass gun control laws. Throughout the years, gun control laws have become more rigid, and groups, both in support of and against gun control, have become more adamant in their stances. Critics contend that gun control laws infringe on the Constitutional right to own guns, while supporters claim that firearms must be strictly regulated in order to reduce gun related violence. Whatever the solution, America must find a way to balance personal freedoms with public safety.
		One of the most persuasive pieces of evidence against gun control rests in the Second Amendment to the Constitution. The Second Amendment states, “A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.” This Amendment is part of the Constitution, the first and foremost authority of American government and the source that the Supreme Court uses to evaluate laws and court cases. Due to the Second Amendment's overriding power, any legislation prohibiting gun ownership conflicts with the plans of the Founding Fathers and is a direct "threat to freedom".
	"""

	print ("Writeprints Static !!")
	features = ws.calculateFeatures(dummy_text, jfe)
	print("Length of features: ", len(features))
	print("Features List: ", features)
