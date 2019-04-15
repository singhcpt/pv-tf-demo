package org.tensorflow.demo.tracking;

/**
 * Created by plantvillage on 1/30/18.
 */

public class Triplet<T, U, V> {
    private final T first;
    private final U second;
    private final V third;

    /**
     * Constructor for a Triplet.
     *
     * @param first the first object in the Triplet
     * @param second the second object in the triplet
     * @param third the second object in the triplet

     */
    public Triplet(T first, U second, V third) {
        this.first = first;
        this.second = second;
        this.third = third;
    }

    public T getFirst() { return first; }
    public U getSecond() { return second; }
    public V getThird() { return third; }

    /**
     * Convenience method for creating an appropriately typed pair.
     * @param t the first object in the triplet
     * @param u the second object in the triplet
     * @param v the second object in the triplet
     * @return a Pair that is templatized with the types of a and b
     */
    public static <T, U, V> Triplet <T, U, V> create(T t, U u, V v) {
        return new Triplet<T, U, V>(t, u, v);
    }

}
