package org.vadere.state.attributes.models;

import java.util.Arrays;
import java.util.List;

import org.vadere.annotation.factories.attributes.ModelAttributeClass;
import org.vadere.state.attributes.Attributes;

@ModelAttributeClass
public class AttributesSIRG extends Attributes {

	private int infectionsAtStart = 0;
	private double infectionRate = 0.01;

	private double recoveryRate = 0.001;
	private double infectionMaxDistance = 1;

	public int getInfectionsAtStart() { return infectionsAtStart; }

	public double getInfectionRate() {
		return infectionRate;
	}

	public double getRecoveryRate() {
		return recoveryRate;
	}

	public double getInfectionMaxDistance() {
		return infectionMaxDistance;
	}

}
