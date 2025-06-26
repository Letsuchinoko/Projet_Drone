package com.example.APPLI

import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import android.widget.ProgressBar

// TODO: Rename parameter arguments, choose names that match
// the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

/**
 * Fragment qui affiche le niveau sonore reçu via un TextView
 * et une jauge (ProgressBar).
 */
class Fragment3 : Fragment() {
    private val TAG = "Fragment3"
    // TODO: Rename and change types of parameters
    private var param1: String? = null
    private var param2: String? = null

    private lateinit var soundTextView: TextView
    private lateinit var soundLevelProgressBar: ProgressBar
    private val sharedDataManager: SharedDataManager by activityViewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        Log.d(TAG, "Fragment3 onCreateView appelé")
        val view = inflater.inflate(R.layout.fragment_3, container, false)

        // --- Initialisation des composants graphiques (TextView et ProgressBar) ---
        soundTextView = view.findViewById(R.id.soundLevelTextView)
        soundLevelProgressBar = view.findViewById(R.id.soundLevelProgressBar)
        soundTextView.text = "Niveau sonore\n\nEn attente de données..."
        soundLevelProgressBar.progress = 0
        Log.d(TAG, "TextView sonore initialisé")

        // --- Observation des données sonores partagées ---
        sharedDataManager.soundData.observe(viewLifecycleOwner) { data ->
            Log.d(TAG, "Données sonores reçues dans Fragment3: '$data'")

            // Regex pour extraire la valeur numérique du niveau sonore
            val soundPattern = Regex("-?\\d+\\.?\\d*")
            val matchResult = soundPattern.find(data)

            if (matchResult != null) {
                // Mise à jour du TextView avec la valeur
                val soundLevel = matchResult.value
                soundTextView.text = "Niveau sonore\n\n$soundLevel dB"
                Log.d(TAG, "Niveau sonore trouvé et affiché: $soundLevel")

                // Mise à jour de la jauge (ProgressBar)
                val soundValue = soundLevel.toFloatOrNull()
                if (soundValue != null) {
                    // Limite la valeur entre 0 et 120 pour la jauge
                    val progress = soundValue.coerceIn(0f, 120f).toInt()
                    soundLevelProgressBar.progress = progress
                }
            }
        }

        Log.d(TAG, "Fragment3 onCreateView terminé")
        return view
    }

    companion object {
        /**
         * Factory method pour créer une nouvelle instance du fragment.
         *
         * @param param1 Parameter 1.
         * @param param2 Parameter 2.
         * @return A new instance of fragment Fragment3.
         */
        // TODO: Rename and change types and number of parameters
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            Fragment3().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}