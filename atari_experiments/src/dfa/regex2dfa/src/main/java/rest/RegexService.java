package rest;

import dk.brics.automaton.Automaton;
import dk.brics.automaton.CustomAutomaton;
import dk.brics.automaton.RegExp;

/**
 * Created by hd on 04/14/18.
 */
public class RegexService {
    private String regex;
    private boolean totalize;
    private String dfaInDot;

    public RegexService(String stateLetter, String regex, boolean totalize) {
        this.regex = regex;
        this.totalize = totalize;
        this.dfaInDot = toDot(stateLetter);
    }

    public String toDot(String stateLetter) {
        RegExp r = new RegExp(regex);
        Automaton automaton = r.toAutomaton();
        automaton.determinize();
        automaton.minimize();

        assert automaton.isDeterministic() == true;

        CustomAutomaton customAutomaton = new CustomAutomaton(automaton);
        if (totalize)
            customAutomaton.totalize();

        String dot = customAutomaton.toDot(stateLetter);

        return dot;
    }

    @Override
    public String toString() {
        return "RegexService{" +
                "regex='" + regex + '\'' +
                ", totalize=" + totalize +
                ", dfaInDot='" + dfaInDot + '\'' +
                '}';
    }
}
