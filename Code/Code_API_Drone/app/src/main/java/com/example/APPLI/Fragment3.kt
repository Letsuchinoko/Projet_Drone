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

// TODO: Rename parameter arguments, choose names that match
// the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

/**
 * A simple [Fragment] subclass.
 * Use the [Fragment3.newInstance] factory method to a
 * create an instance of this fragment.
 */
class Fragment3 : Fragment() {
    private val TAG = "Fragment3"
    // TODO: Rename and change types of parameters
    private var param1: String? = null
    private var param2: String? = null

    private lateinit var soundTextView: TextView
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

        // Initialiser la vue
        soundTextView = view.findViewById(R.id.soundLevelTextView)
        soundTextView.text = "Niveau sonore\n\nEn attente de données..."
        Log.d(TAG, "TextView sonore initialisé")

        // Observer les données sonores et extraire la valeur numérique
        sharedDataManager.soundData.observe(viewLifecycleOwner) { data ->
            Log.d(TAG, "Données sonores reçues dans Fragment3: '$data'")

            // Regex pour trouver un nombre (peut être précédé de "SOUND:")
            val soundPattern = Regex("-?\\d+\\.?\\d*")
            val matchResult = soundPattern.find(data)

            if (matchResult != null) {
                val soundLevel = matchResult.value
                soundTextView.text = "Niveau sonore\n\n$soundLevel dB"
                Log.d(TAG, "Niveau sonore trouvé et affiché: $soundLevel")
            }
            // Si aucun nombre n'est trouvé, on ne met pas à jour l'affichage
        }

        Log.d(TAG, "Fragment3 onCreateView terminé")
        return view
    }

    companion object {
        /**
         * Use this factory method to create a new instance of
         * this fragment using the provided parameters.
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