import dk.brics.automaton.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Main {

    public static void main(String[] args) {

        if(args.length != 2){
            System.out.println("Usage: program regular_expression file");
            System.exit(1);
        }
        String regExString = args[0];
        String fileName = args[1];


        RegExp re = new RegExp(regExString);
        Automaton dfa = re.toAutomaton();
        dfa.minimize();


        int numStates = dfa.getNumberOfStates();
        char transitionMatrix[][] = new char[numStates][];
        for(int i=0; i<numStates; i++)
            transitionMatrix[i] = new char[numStates];
        for(int i=0; i<numStates; i++){
            for(int j=0; j<numStates; j++) {
                transitionMatrix[i][j] = '_';
            }
        }

        Set<State> states = dfa.getStates();
        for(State s: states) {
            int startingStateName = new Integer(getStateName(s));
            List<Transition> trans = s.getSortedTransitions(true);
            for(Transition t: trans){
                int targetStateName = new Integer(getTargetStateName(t));
                char symbol = t.getMin();
                transitionMatrix[startingStateName][targetStateName] = symbol;
            }
        }

        try {
            FileWriter fileWriter = new FileWriter(fileName);
            BufferedWriter bw = new BufferedWriter(fileWriter);

            bw.write(getStateName(dfa.getInitialState()) + "\n");

            int i = 0;
            List<State> acceptStates = new ArrayList<>(dfa.getAcceptStates());
            for(;i < acceptStates.size()-1; i++)
                bw.write(getStateName(acceptStates.get(i)) + " ");
            bw.write(getStateName(acceptStates.get(i)) + "\n");

            for(i=0; i<numStates; i++){
                //bw.write(i+" ");
                int j;
                for(j=0; j<numStates-1; j++){
                    bw.write(transitionMatrix[i][j] + " ");
                }
                bw.write(transitionMatrix[i][j]);
                bw.write("\n");
            }

            bw.close();
        } catch (IOException e){
            System.out.println("Unable to write to file: " + fileName);
        }

    }

    static String getStateName(State s){
        return s.toString().split(" ")[1];
    }

    static String getTargetStateName(Transition t){
        return t.toString().split(" ")[2];
    }

}
