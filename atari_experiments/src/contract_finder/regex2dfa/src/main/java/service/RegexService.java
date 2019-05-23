package service;

import dk.brics.automaton.Automaton;
import dk.brics.automaton.CustomAutomaton;
import dk.brics.automaton.RegExp;
import util.FileUtil;

import java.io.IOException;

/**
 * Created by hd on 04/14/18.
 */
public class RegexService {
    private boolean totalize;
    private String regex;

    public RegexService(boolean totalize, String path) throws IOException {
        this.totalize = totalize;
        this.regex = FileUtil.readAll(path);
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
}
