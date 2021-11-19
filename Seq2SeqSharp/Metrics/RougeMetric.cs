using System;
using System.Collections.Generic;

using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Metrics
{
    public class RougeMetric : IMetric
    {
        public string Name => "RougeL";

        readonly HashSet<string> allRefWords = new HashSet<string>();
        readonly HashSet<string> allHypWords = new HashSet<string>();
        readonly HashSet<string> unionLCSWords = new HashSet<string>();

        public void ClearStatus()
        {
            allRefWords.Clear();
            allHypWords.Clear();
            unionLCSWords.Clear();
        }

        public void Evaluate( List<List<string>> refTokens, List<string> hypTokens )
        {
            var refSent = refTokens[ 0 ];

            foreach ( var token in refSent )
            {
                allRefWords.Add( token );
            }

            foreach ( var token in hypTokens )
            {
                allHypWords.Add( token );
            }

            var results = Lcs( refSent, hypTokens );
            foreach ( var r in results )
            {
                unionLCSWords.Add( r );
            }

        }

        public double GetPrimaryScore()
        {
            double recall = unionLCSWords.Count / (double) allRefWords.Count;
            double precision = unionLCSWords.Count / (double) allHypWords.Count;
            double objective = 0.0;
            if ( precision > 0.0 && recall > 0.0 )
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }

            return objective;
        }

        public string GetScoreStr()
        {
            return GetPrimaryScore().ToString( "F" );
        }

        static List<string> Lcs( List<string> X, List<string> Y )
        {
            int m = X.Count;
            int n = Y.Count;

            int[,] L = new int[ m + 1, n + 1 ];

            // Following steps build L[m+1][n+1] in bottom up fashion. Note 
            // that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]  
            for ( int i1 = 0; i1 <= m; i1++ )
            {
                for ( int j1 = 0; j1 <= n; j1++ )
                {
                    if ( i1 == 0 || j1 == 0 )
                        L[ i1, j1 ] = 0;
                    else if ( X[ i1 - 1 ].Equals( Y[ j1 - 1 ] ) )
                        L[ i1, j1 ] = L[ i1 - 1, j1 - 1 ] + 1;
                    else
                        L[ i1, j1 ] = Math.Max( L[ i1 - 1, j1 ], L[ i1, j1 - 1 ] );
                }
            }

            // Following code is used to print LCS 
            int index = L[ m, n ];
            int temp = index;

            // Create a character array to store the lcs string 
            string[] lcs = new string[ index + 1 ];
            lcs[ index ] = ""; // Set the terminating character 

            // Start from the right-most-bottom-most corner and 
            // one by one store characters in lcs[] 
            int i = m, j = n;
            while ( i > 0 && j > 0 )
            {
                // If current character in X[] and Y are same, then 
                // current character is part of LCS 
                if ( X[ i - 1 ].Equals( Y[ j - 1 ] ) )
                {
                    // Put current character in result 
                    lcs[ index - 1 ] = X[ i - 1 ];

                    // reduce values of i, j and index 
                    i--;
                    j--;
                    index--;
                }

                // If not same, then find the larger of two and 
                // go in the direction of larger value 
                else if ( L[ i - 1, j ] > L[ i, j - 1 ] )
                    i--;
                else
                    j--;
            }

            var al = new List<string>( temp );
            // Print the lcs 

            for ( int k = 0; k <= temp; k++ )
                if ( !lcs[ k ].IsNullOrEmpty() )
                {
                    al.Add( lcs[ k ] );
                }

            return al;
        }
    }
}
