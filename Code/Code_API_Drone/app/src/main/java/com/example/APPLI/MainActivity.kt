package com.example.APPLI

import android.os.Bundle
import android.util.Log
import android.widget.Button
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import com.google.android.material.navigation.NavigationView
import androidx.appcompat.widget.Toolbar

class MainActivity : AppCompatActivity() {
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var navigationView: NavigationView
    private var currentFragment: Fragment? = null

    private fun remplaceFragment(fragment: Fragment) {
        Log.d("MainActivity", "Remplacement du fragment: ${fragment.javaClass.simpleName}")
        val fragMan: FragmentManager = supportFragmentManager
        val fragTran = fragMan.beginTransaction()
        if (currentFragment != null) {
            fragTran.hide(currentFragment!!)
        }
        if (!fragment.isAdded) {
            fragTran.add(R.id.fragmentContainerView, fragment)
        } else {
            fragTran.show(fragment)
        }
        fragTran.addToBackStack(null)
        fragTran.commit()
        currentFragment = fragment
    }

    override fun onBackPressed() {
        val fragMan = supportFragmentManager
        if (fragMan.backStackEntryCount > 1) {
            fragMan.popBackStack()
            // Met à jour le currentFragment avec le fragment précédent
            val fragments = fragMan.fragments
            for (i in fragments.size - 1 downTo 0) {
                val frag = fragments[i]
                if (frag.isVisible) {
                    currentFragment = frag
                    break
                }
            }
        } else {
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        // Initialize fragments
        val frag1 = Fragment1()
        val frag2 = Fragment2()
        val frag3 = Fragment3()

        // Setup toolbar
        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.setHomeAsUpIndicator(android.R.drawable.ic_menu_sort_by_size)

        // Setup navigation drawer
        drawerLayout = findViewById(R.id.main)
        navigationView = findViewById(R.id.nav_view)

        // Handle navigation item clicks
        navigationView.setNavigationItemSelectedListener { menuItem ->
            Log.d("MainActivity", "Menu item clicked: ${menuItem.title}")
            when (menuItem.itemId) {
                R.id.item1 -> {
                    remplaceFragment(frag1)
                    drawerLayout.closeDrawers()
                    true
                }
                R.id.item2 -> {
                    remplaceFragment(frag2)
                    drawerLayout.closeDrawers()
                    true
                }
                R.id.item3 -> {
                    remplaceFragment(frag3)
                    drawerLayout.closeDrawers()
                    true
                }
                else -> {
                    Log.d("MainActivity", "Unknown menu item clicked")
                    false
                }
            }
        }

        // Setup buttons
        val bouton1 = findViewById<Button>(R.id.button)
        val bouton2 = findViewById<Button>(R.id.button2)

        // Set initial fragment
        remplaceFragment(frag1)

        bouton1.setOnClickListener {
            Log.d("MainActivity", "Bouton Arrière clicked")
            when (currentFragment) {
                is Fragment2 -> remplaceFragment(frag1)
                is Fragment3 -> remplaceFragment(frag2)
                else -> {} // Already at Fragment1
            }
        }

        bouton2.setOnClickListener {
            Log.d("MainActivity", "Bouton Suivante clicked")
            when (currentFragment) {
                is Fragment1 -> remplaceFragment(frag2)
                is Fragment2 -> remplaceFragment(frag3)
                else -> {} // Already at Fragment3
            }
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        Log.d("MainActivity", "Navigation up clicked")
        drawerLayout.openDrawer(GravityCompat.START)
        return true
    }
}