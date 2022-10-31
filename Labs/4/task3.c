#include <stdio.h>
#include <stdint.h>     // provides int8_t, uint8_t, int16_t etc.
#include <stdlib.h>

struct a
{
    int64_t x;
    int32_t y, z;
};

struct b
{
    int32_t x; //, z;
    int64_t y;
    int32_t z;
};

int main(void)
{
    int32_t i1;
    int64_t l1;
    struct a a1;
    struct b b1;
    int32_t i2, i3, i4, i5;
    int64_t l2, l3, l4, l5;
    struct a a2, a3, a4, a5;
    struct b b2, b3, b4, b5;

    // TODO
    // afisati pe cate o linie adresele lui i1, l1, a1, b1
    // apoi i2, l2, a2, b2 etc.
    // HINT folositi %p pentru a afisa o adresa
    // ce observati?
    
    printf("Variables' addresses' offsets are:\n");
    
    //printf("%p %p %p %p\n%p %p %p %p\n%p %p %p %p\n%p %p %p %p\n%p %p %p %p\n",
    //	    &i1, &l1, &a1, &b1, &i2, &l2, &a2, &b2, &i3, &l3, &a3, &b3, &i4, &l4, &a4, &b4, &i5, &l5, &a5, &b5);
    
    printf("+%lu +%lu +%lu +%lu\n+%lu +%lu +%lu +%lu\n+%lu +%lu +%lu +%lu\n+%lu +%lu +%lu +%lu\n+%lu +%lu +%lu +%lu\n\n",
    	    (void*)(&i1) - (void*)(&i1), (void*)(&l1) - (void*)(&i1), (void*)(&a1) - (void*)(&i1), (void*)(&b1) - (void*)(&i1),
    	    (void*)(&i2) - (void*)(&i1), (void*)(&l2) - (void*)(&i1), (void*)(&a2) - (void*)(&i1), (void*)(&b2) - (void*)(&i1),
    	    (void*)(&i3) - (void*)(&i1), (void*)(&l3) - (void*)(&i1), (void*)(&a3) - (void*)(&i1), (void*)(&b3) - (void*)(&i1),
    	    (void*)(&i4) - (void*)(&i1), (void*)(&l4) - (void*)(&i1), (void*)(&a4) - (void*)(&i1), (void*)(&b4) - (void*)(&i1),
    	    (void*)(&i5) - (void*)(&i1), (void*)(&l5) - (void*)(&i1), (void*)(&a5) - (void*)(&i1), (void*)(&b5) - (void*)(&i1));

    // TODO
    // ce dimensiune credeti ca au structurile a si b?
    // afisati dimensiunea lor folosind operatorul sizeof
    // explicati rezultatul
    printf("sizeof(struct a) = %lu, sizeof(struct b) = %lu\n\n", sizeof(struct a), sizeof(struct b));

    // TODO
    // alocati un vector cu 10 elemente de tip float aliniat la 2^5 bytes
    
    // REZOLVARE
    // 2^5 = 32
    // deoarece nu stim ce aliniere returneaza malloc vom presupune ca nu
    // aliniaza
    // in aceasta situatie, cel mai defavorabil caz apare cand primim o adresa
    // de forma (32 * k + 1)
    // vom avea nevoie de 31 de bytes de padding pentru a obtine alinierea
    // dorita
    void* aux_vect = malloc(10 * sizeof(float) + 31);
    // adresa obtinuta trebuie transformata intr-o adresa aliniata la 32 bytes
    // care pointeaza in interiorul buffer-ului alocat
    // alinierea unei valori la multiplu de 32 se face cu & folosind
    // masca ~31 (binar: 1111...11100000)
    // deoarece rezultatul acestei operatii este tot timpul mai mic decat adresa
    // initiala si ne-ar scoate in afara buffer-ului alocat, vom aduna 31 la
    // adresa initiala inainte de mascare
    // verificati pe hartie ca aceasta operatie returneaza tot timpul o adresa
    // in interiorul buffer-ului, aliniata la 32 bytes, iar spatiul irosit cu
    // padding-ul este minim posibil
    // aux_vect este pastrat pentru a putea apela free cu pointer-ul original
    // returnat de malloc
    // uintptr_t este un tip integer de marime suficienta pentru a pastra
    // adresa necesara pentru a face operatii aritmetice cu adresa nealiniata
    float* vect = (float*)(((uintptr_t)aux_vect + 31) & ~31);

    printf("aux_vect: %p\n", aux_vect);
    printf("vect:     %p\n", vect);
    
    printf("aux_vect: %lu\n", &aux_vect);
    printf("vect:     %lu\n", &vect);

    // trebuie se eliberam memoria folosind adresa intiala, returnata de malloc
    free(aux_vect);

    return 0;
}

