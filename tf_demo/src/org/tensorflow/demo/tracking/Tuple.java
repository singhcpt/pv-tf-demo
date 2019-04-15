package org.tensorflow.demo.tracking;

/**
 * Created by plantvillage on 1/30/18.
 */

public class Tuple<T, U> {
    private final T first;
    private final U second;

    /**
     * Constructor for a Triplet.
     *
     * @param first the first object in the tuple
     * @param second the second object in the tuple

     */
    public Tuple(T first, U second) {
        this.first = first;
        this.second = second;
    }

    public T getFirst() { return first; }
    public U getSecond() { return second; }

    /**
     * Convenience method for creating an appropriately typed pair.
     * @param t the first object in the tuple
     * @param u the second object in the tuple
     * @return a Pair that is templatized with the types of a and b
     */
    public static <T, U> Tuple <T, U> create(T t, U u) {
        return new Tuple<T, U>(t, u);
    }

}